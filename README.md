# TensorFlow Federated Aggregation
Using [TF Encrypted](https://github.com/tf-encrypted/tf-encrypted) primitives for secure aggregation in [TensorFlow Federated](https://github.com/tensorflow/federated)

This project implements specific protocols for secure aggregation using secure computation primitives from [TF Encrypted](https://github.com/tf-encrypted/tf-encrypted). Our aim is to express secure aggregations with the full breadth of TFE's language for secure computations, however this prototype is much smaller in scope. We implement a specific aggregation protocol based on Paillier homomorphic encryption; see the [accompanying blog post](https://medium.com/dropoutlabs/building-secure-aggregation-into-tensorflow-federated-4514fca40cc0) or the sections below for more details.

```python
import tensorflow as tf
import tensorflow_federated as tff
from federated_aggregations import paillier

paillier_factory = paillier.local_paillier_executor_factory()
tff.framework.set_default_executor(paillier_factory)

# data from 5 clients
x = [np.array([i, i + 1], dtype=np.int32) for i in range(5)]
x_type = tff.TensorType(tf.int32, [2])

@tff.federated_computation(tff.FederatedType(x_type, tff.CLIENTS))
def secure_paillier_addition(x):
  return tff.federated_secure_sum(x, bitwidth=32)

result = secure_paillier_addition(x)
print(result)
>>> [10 15]
```

# Installation
This library is offered as a Python package but is not currently published on PyPI, so you must install it from source in your preferred Python environment.

```
pip install -r requirements.txt
python setup.py install
```

# Features
## Federated computation
Currently, we simply add an implementation of the [`tff.federated_secure_sum`](https://www.tensorflow.org/federated/api_docs/python/tff/federated_secure_sum) to the default TFF simulation stack. The complete protocol is visualized below (TODO). We do not rewrite any of the higher-level APIs for federated averaging, but these should be straightforward to implement.

## Protocols
Currently, we implement secure aggregation via the Paillier homomorphic encryption scheme. This protocol is well suited for federated averaging with highly-available clients, e.g. cross-silo federated learning between organizations. Please see the [accompanying blog post](https://medium.com/dropoutlabs/building-secure-aggregation-into-tensorflow-federated-4514fca40cc0) for an illustration and more details.

We outsource the Paillier aggregation to an "aggregation service" running separately from the Server role in traditional FL. This fits into the bulletin-board style of FL, where a service separate from the coordinator is responsible for aggregating model updates securely, and the Server (i.e. coordinator) periodically pulls & decrypts the latest model from that service. This achieves the specific functionality outlined in [this section](https://github.com/tf-encrypted/rfcs/tree/master/20190924-tensorflow-federated#specific-encrypted-executors) of the corresponding RFC.

## Secure Channels
TensorFlow Federated does not implement point-to-point communication directly between placements; it instead routes all communications through a computation "driver" (i.e. the host running the TFF Python script, also usually responsible for unplaced computation). To reduce communication when using the native backend, this driver is usually collocated with the tff.SERVER placement's executor stack, so that any values communicated between the driver and the tff.SERVER don't incur a network cost.

This communication pattern presents a problem for implementing secure aggregation, since many SMPC protocols assume the existence of authenticated channels between parties. In order to realize this in the specific case of a bulletin-board aggregation service, we follow the approach outlined in [this section](https://github.com/tf-encrypted/rfcs/tree/master/20190924-tensorflow-federated#network-strategy-and-secure-channels) of our RFC. Please see the [accompanying blog post](https://medium.com/dropoutlabs/building-secure-aggregation-into-tensorflow-federated-4514fca40cc0) for an illustration and more details.

# Development
If you want to get up and running, please follow these steps. We strongly encourage using a virtual environment.
1. Install dependencies with `pip install -r requirements.txt`. Depending on your platform, you may need to build these projects from source. See instructions specific to [tf-encrypted-primitives](https://github.com/tf-encrypted/tf-encrypted/tree/master/primitives) or [tf-big](https://github.com/tf-encrypted/tf-big) for more information. We do not guarantee support for all platforms.
2. Install this package using pip (e.g. `pip install -e .`).
3. Run tests.

If you run into issues, please [reach out](#support-and-feedback).

# Roadmap
Please see the original [TFF Integration RFC](https://github.com/tf-encrypted/rfcs/tree/master/20190924-tensorflow-federated) for an overview of our goals. While the implementation in this project isn't identical, and our plans have evolved since then, the high-level objectives have not changed.

# Support and Feedback
Bug reports and feature requests? Please [open an issue](https://github.com/tf-encrypted/federated-aggregations/issues) on Github.

For any other questions or feedback, please reach out directly on [Slack](https://join.slack.com/t/tf-encrypted/shared_invite/enQtNjI5NjY5NTc0NjczLWM4MTVjOGVmNGFkMWU2MGEzM2Q5ZWFjMTdmZjdmMTM2ZTU4YjJmNTVjYmE1NDAwMDIzMjllZjJjMWNiMTlmZTQ), or send an email to [contact@tf-encrypted.io](mailto:contact@tf-encrypted.io).

# License

Licensed under Apache License, Version 2.0 (see [LICENSE](./LICENSE) or http://www.apache.org/licenses/LICENSE-2.0). Copyright as specified in [NOTICE](./NOTICE).
