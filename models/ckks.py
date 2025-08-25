import tenseal as ts
import time
import numpy as np
import sys
import math
import hashlib

# Context 生成与分发（密钥都在 Context 里面）
# 分发过程需要序列化操作
class TensealKeyManage:
    def __init__(self):
        self.ckks_context = ts.context(ts.SCHEME_TYPE.CKKS, 8192,
                                       coeff_mod_bit_sizes=[60, 40, 40, 60])
        self.ckks_context.global_scale = pow(2, 40)
        self.ckks_context.generate_relin_keys()
    # 分发 CKKS 公钥
    def get_ckks_publicKey(self) -> bytes:
        return self.ckks_context.serialize(save_public_key=True,
                                           save_secret_key=False,
                                           save_relin_keys=True,
                                           save_galois_keys=False)

    # 分发 CKKS 所有密钥
    def get_ckks_secretKey(self) -> bytes:
        return self.ckks_context.serialize(save_public_key=False,
                                           save_secret_key=True,
                                           save_relin_keys=False,
                                           save_galois_keys=False)



def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))  # 计算大小的单位索引
    p = math.pow(1024, i)  # 计算转换的比例
    s = round(size_bytes / p, 2)  # 得到转换后的结果
    return "%s %s" % (s, size_name[i])  # 返回格式化后的字符串




if __name__ == '__main__':
    # 密钥管理器
    n = 4096
    k = 5
    a = np.random.uniform(-3, -2, size=n)
    b = np.random.uniform(-4, -5, size=n)
    print(a)
    print(b)
    st = time.time()
    keyManage = TensealKeyManage()
    ed = time.time()
    print(f"Key Gen Time: {(ed -st) * 1000} ms")
    public_key = keyManage.get_ckks_publicKey()
    print(f"public_key_size:{convert_size(len(public_key))}")
    private_key = keyManage.get_ckks_secretKey()
    print(f"private_key_size:{convert_size(len(private_key))}")

    enc_a = ts.ckks_vector(keyManage.ckks_context, a)
    enc_b = ts.ckks_vector(keyManage.ckks_context, b)
    enc_sum = enc_b - enc_a
    print(f"解密的密文和 {enc_sum.decrypt()}")
