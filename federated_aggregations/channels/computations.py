import tensorflow_federated as tff
from tf_encrypted.primitives.sodium import easy_box


def make_encryptor(plaintext_type, pk_rcv_type, sk_snd_type):
  @tff.tf_computation(plaintext_type, pk_rcv_type, sk_snd_type)
  def encrypt_tensor(plaintext, pk_rcv, sk_snd):
    pk_rcv = easy_box.PublicKey(pk_rcv)
    sk_snd = easy_box.SecretKey(sk_snd)
    nonce = easy_box.gen_nonce()
    ciphertext, mac = easy_box.seal_detached(plaintext, nonce, pk_rcv, sk_snd)
    return ciphertext.raw, mac.raw, nonce.raw

  return encrypt_tensor


def make_decryptor(sender_values_type, pk_snd_type, sk_rcv_snd,
    orig_tensor_dtype):
  @tff.tf_computation(sender_values_type, pk_snd_type, sk_rcv_snd)
  def decrypt_tensor(sender_values, pk_snd, sk_rcv):
    ciphertext = easy_box.Ciphertext(sender_values[0])
    mac = easy_box.Mac(sender_values[1])
    nonce = easy_box.Nonce(sender_values[2])
    pk_snd = easy_box.PublicKey(pk_snd)
    sk_rcv = easy_box.SecretKey(sk_rcv)
    plaintext_recovered = easy_box.open_detached(
        ciphertext, mac, nonce, pk_snd, sk_rcv, orig_tensor_dtype)
    return plaintext_recovered

  return decrypt_tensor


def make_keygen():
  @tff.tf_computation()
  def key_generator():
    pk, sk = easy_box.gen_keypair()
    return pk.raw, sk.raw

  return key_generator
