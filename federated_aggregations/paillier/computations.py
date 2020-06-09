import tensorflow as tf
import tensorflow_federated as tff
import tf_big
from tf_encrypted.primitives import paillier


def make_keygen(bitlength=2048):
  @tff.tf_computation
  def _keygen():
    encryption_key, decryption_key = paillier.gen_keypair(bitlength)
    ek_raw = encryption_key.export(dtype=tf.string)
    dk_raw = decryption_key.export(dtype=tf.string)
    return ek_raw, dk_raw

  return _keygen


def make_encryptor():
  @tff.tf_computation
  def _encrypt(encryption_key_raw, plaintext):
    ek = paillier.EncryptionKey(encryption_key_raw)
    ciphertext = paillier.encrypt(ek, plaintext)
    return ciphertext.export(dtype=tf.string)

  return _encrypt


def make_decryptor(
    decryption_key_type, encryption_key_type, ciphertext_type, dtype):
  @tff.tf_computation(
      decryption_key_type, encryption_key_type, ciphertext_type)
  def _decrypt(decryption_key_raw, encryption_key_raw, ciphertext_raw):
    dk = paillier.DecryptionKey(*decryption_key_raw)
    ek = paillier.EncryptionKey(encryption_key_raw)
    c = tf_big.convert_to_tensor(ciphertext_raw)
    ciphertext = paillier.Ciphertext(ek, c)
    return paillier.decrypt(dk, ciphertext, dtype)

  return _decrypt


def make_sequence_sum():
  def adder(ek, xs):
    assert len(xs) >= 1
    if len(xs) == 1:
      return xs[0]
    split = len(xs) // 2
    left = xs[:split]
    right = xs[split:]
    return paillier.add(ek, adder(ek, left), adder(ek, right), do_refresh=False)

  @tff.tf_computation
  def _sequence_sum(encryption_key_raw, summands_raw):
    ek = paillier.EncryptionKey(encryption_key_raw)
    summands = [paillier.Ciphertext(ek, summand) for summand in summands_raw]
    result = adder(ek, summands)
    refreshed_result = paillier.refresh(ek, result)
    return refreshed_result.export(dtype=tf.string)

  return _sequence_sum
