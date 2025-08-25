from models.ckks import TensealKeyManage
import phe.paillier as paillier


class TA:
    def __init__(self):
        self.ckks_keygen = TensealKeyManage()
        self.ckks_public_key, self.ckks_secret_key = self.gen_ckks_key()
        self.paillier_public_key, self.paillier_secret_key = self.gen_paillier_key()


    def gen_ckks_key(self):
        ckks_secret_key = self.ckks_keygen.get_ckks_secretKey()
        ckks_public_key = self.ckks_keygen.get_ckks_publicKey()
        return ckks_public_key, ckks_secret_key

    def gen_paillier_key(self):
        paillier_public_key, paillier_secret_key = paillier.generate_paillier_keypair(n_length=128)
        return paillier_public_key, paillier_secret_key
