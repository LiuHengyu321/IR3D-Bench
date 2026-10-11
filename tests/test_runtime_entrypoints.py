"""Offline entry-point regression tests; model operations are stubbed."""
import argparse
import ast
import glob
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def load_functions(relative_path, names, namespace):
    # Execute the real entry-point functions without importing GPU dependencies.
    path = ROOT / relative_path
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace['__file__'] = str(path)
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


class RuntimeTests(unittest.TestCase):
    def test_api_small_sample_and_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / 'images'
            images.mkdir()
            for index in range(3):
                (images / f'CLEVR_val_{index:06d}.png').touch()
            prompt = root / 'prompt.txt'
            prompt.write_text('reconstruct')
            output = root / 'json'
            output.mkdir()
            existing = output / 'CLEVR_val_000000.json'
            existing.write_text('{"existing": true}')
            client = Mock()
            client.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"objects": []}'))])
            factory = Mock(return_value=client)
            namespace = load_functions('main_api.py', ['main'], {
                'argparse': argparse, 'os': os, 'glob': glob, 'json': json,
                'OpenAI': factory, 'Image': Mock(),
                'encode_image': lambda image: 'image',
                'build_message_with_image_and_text': lambda text, image: [],
                'tqdm': lambda values, **kwargs: values,
            })
            argv = ['main_api.py', '--image_dir', str(images), '--result_dir', str(output),
                    '--prompt_path', str(prompt), '--model_name', 'test-model']
            with patch.object(sys, 'argv', argv), patch.dict(os.environ, {
                'OPENAI_API_KEY': 'offline-test', 'OPENAI_BASE_URL': 'https://example.invalid/v1',
            }):
                namespace['main']()
            self.assertEqual(client.chat.completions.create.call_count, 2)
            self.assertEqual(len(list(output.glob('*.json'))), 3)
            self.assertEqual(json.loads(existing.read_text()), {'existing': True})
            factory.assert_called_once_with(api_key='offline-test', base_url='https://example.invalid/v1')

    def test_vllm_paths_limit_and_default_all(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / 'images'
            images.mkdir()
            for index in range(22):
                (images / f'CLEVR_val_{index:06d}.png').touch()
            request = SimpleNamespace(engine_args={}, stop_token_ids=None,
                                      prompts=['prompt'], lora_requests=None)
            llm = Mock()
            llm.generate.return_value = [SimpleNamespace(outputs=[SimpleNamespace(text='{}')])]
            runner = Mock()
            namespace = load_functions('main_vllm.py', ['parse_args', 'main'], {
                'FlexibleArgumentParser': argparse.ArgumentParser,
                'os': os, 'glob': glob, 're': re, 'json': json,
                'model_example_map': {'test-model': lambda *args: request},
                'asdict': lambda value: value, 'LLM': Mock(return_value=llm),
                'SamplingParams': lambda **kwargs: kwargs, 'Image': Mock(),
                'extract_nth_json_object': lambda value: value,
                'subprocess': SimpleNamespace(run=runner),
            })
            for limit, expected in [(None, 22), (1, 1)]:
                llm.reset_mock()
                argv = ['main_vllm.py', '--model-type', 'test-model', '--image-dir', str(images),
                        '--output-dir', 'relative output', '--blender-path', '/custom/blender']
                if limit:
                    argv += ['--max-images', str(limit)]
                with patch.object(sys, 'argv', argv):
                    args = namespace['parse_args']()
                # Keep output paths relative while avoiding changes to the checkout.
                previous = os.getcwd()
                try:
                    os.chdir(root)
                    namespace['main'](args)
                finally:
                    os.chdir(previous)
                self.assertEqual(llm.generate.call_count, expected)
                self.assertTrue((root / 'relative output/json/CLEVR_val_000000.json').exists())
                command = runner.call_args.args[0]
                self.assertEqual(command[0], '/custom/blender')
                self.assertEqual(command[3], str(ROOT / 'pipeline/recon.py'))
                self.assertEqual(command[-2:], ['relative output/json', 'relative output/render'])

    def test_metrics_help_and_missing_checkpoint_do_not_load_models(self):
        loaders = Mock()
        namespace = load_functions('pipeline/metrics.py', ['main'], {
            'argparse': argparse, 'os': os, 'sys': sys,
            'CLIPTokenizer': loaders, 'CLIPModel': loaders,
            'sam_model_registry': Mock(),
        })
        required = [item for name in ['map_dir', 'render_dir', 'gt_json_dir', 'pred_json_dir',
                                     'gt_image_dir', 'output_dir'] for item in ['--' + name, '.']]
        for extra, code in [(['--help'], 0), (required + ['--sam_checkpoint', '/missing/checkpoint'], 2)]:
            with patch.object(sys, 'argv', ['metrics.py'] + extra):
                with self.assertRaises(SystemExit) as error:
                    namespace['main']()
                self.assertEqual(error.exception.code, code)
        loaders.from_pretrained.assert_not_called()

    def test_eval_script_paths_and_stop_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stub = root / 'stub tool'
            stub.write_text('#!' + sys.executable + '\n'
                            'import json, os, sys\n'
                            'with open(os.environ["CALL_LOG"], "a") as f:\n'
                            '    f.write(json.dumps(sys.argv[1:]) + "\\n")\n'
                            'sys.exit(int(os.environ.get("STUB_EXIT", "0")))\n')
            stub.chmod(0o755)
            checkpoint = root / 'sam checkpoint.pth'
            checkpoint.touch()
            log = root / 'calls.jsonl'
            env = dict(os.environ, PYTHON_BIN=str(stub), BLENDER_PATH=str(stub),
                       SAM_CHECKPOINT=str(checkpoint), OPENAI_API_KEY='offline-test', CALL_LOG=str(log))
            command = ['bash', str(ROOT / 'cal_metric.sh'), 'output dir', 'gt dir', '0']
            result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[0][0], str(ROOT / 'pipeline/match_fast.py'))
            self.assertEqual(calls[1][2], str(ROOT / 'pipeline/recon.py'))
            self.assertEqual(calls[2][0], str(ROOT / 'pipeline/metrics.py'))
            self.assertIn('output dir/json', calls[0])
            self.assertIn('gt dir/json', calls[0])
            self.assertNotIn('--openai_api_key', calls[2])
            log.unlink()
            result = subprocess.run(command, cwd=root, env=dict(env, STUB_EXIT='7'), capture_output=True)
            self.assertEqual(result.returncode, 7)
            self.assertEqual(len(log.read_text().splitlines()), 1)


if __name__ == '__main__':
    unittest.main()
