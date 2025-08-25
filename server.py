from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import threading
import os

app = Flask(__name__)
CORS(app)

is_running = False  # 互斥，避免重复启动

def stream_output(pipe, prefix=b""):
    # 二进制逐行读取 + UTF-8 解码，防止中文乱码
    try:
        for b in iter(pipe.readline, b""):
            print((prefix + b).decode('utf-8', errors='replace'), end='', flush=True)
    finally:
        try:
            pipe.close()
        except Exception:
            pass

@app.route('/run-demo', methods=['POST'])
def run_demo():
    global is_running
    if is_running:
        return jsonify({
            'status': 'running',
            'message': '任务已在运行中，请稍后再试。'
        }), 200

    try:
        # 1) 读取前端 POST JSON
        payload = request.get_json(silent=True) or {}

        # 2) 参数解析与默认值
        def to_bool(v):
            if isinstance(v, bool): return v
            if isinstance(v, (int, float)): return bool(v)
            if isinstance(v, str): return v.lower() in ('1', 'true', 'yes', 'y', 't')
            return False

        dataset = str(payload.get('dataset', 'mnist')).lower()  # 默认 mnist
        security = str(payload.get('security', 'ckks')).lower()
        model = str(payload.get('model', 'cnn')).lower()  # 新增 model，默认 cnn
        epochs = payload.get('epochs', None)
        iid_val = payload.get('iid', True)  # 默认 True（即默认加 --iid）
        num_users = payload.get('num_users', None)
        frac = payload.get('frac', None)

        # 2.1) 简单校验 model 合法性
        if model not in ('cnn', 'resnet18'):
            return jsonify({'status': 'error', 'message': 'invalid model, expected "cnn" or "resnet18"'}), 400
        if dataset not in ('mnist', 'cifar', 'fmnist'):
            return jsonify({'status': 'error', 'message': 'invalid dataset'}), 400
        if security not in ('ckks', 'paillier', 'no'):
            return jsonify({'status': 'error', 'message': 'invalid security'}), 400

        # 3) 组装 main.py 命令行参数（等号形式 argparse 支持）
        demo_args = [
            f'--dataset={dataset}',
            f'--security={security}',
            f'--model={model}',  # 加上 model
        ]

        if epochs is not None:
            demo_args.append(f'--epochs={int(epochs)}')

        # --iid 为 store_true：True => 添加 --iid；False => 不添加
        if to_bool(iid_val):
            demo_args.append('--iid')

        if num_users is not None:
            demo_args.append(f'--num_users={int(num_users)}')
        if frac is not None:
            demo_args.append(f'--frac={float(frac)}')

        # 4) 启动子进程
        py_path = r'D:\Application\Anaconda3\envs\FLHE-test\python.exe'  # 修改为实际路径
        demo_path = os.path.abspath('main.py')
        args = [py_path, '-u', demo_path, *demo_args]

        # 子进程环境：无缓冲 + UTF-8
        child_env = dict(os.environ)
        child_env["PYTHONUNBUFFERED"] = "1"
        child_env["PYTHONIOENCODING"] = "utf-8"

        def run_in_background():
            global is_running
            is_running = True
            try:
                p = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                    env=child_env
                )
                t_out = threading.Thread(target=stream_output, args=(p.stdout, b'[OUT] '), daemon=True)
                t_err = threading.Thread(target=stream_output, args=(p.stderr, b'[ERR] '), daemon=True)
                t_out.start(); t_err.start()
                rc = p.wait()
                t_out.join(); t_err.join()
                print(f'线程执行后端程序运行结束! return_code={rc}', flush=True)
            finally:
                is_running = False

        threading.Thread(target=run_in_background, daemon=True).start()

        return jsonify({
            'status': 'started',
            'message': '已启动！请到后端终端查看实时日志。'
        }), 200

    except Exception as e:
        is_running = False
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # 关闭重载器避免重复拉起子进程；threaded=True 允许并发访问
    app.run(debug=True, port=5000, use_reloader=False, threaded=True)