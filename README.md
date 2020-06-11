# federated-aggregations
Using TF Encrypted primitives for secure aggregation in TensorFlow Federated

This project implements specific protocols for secure aggregation using secure computation primitives from TF Encrypted. The project's aim is to be able to express secure aggregations with the full breadth of TFE's language for secure computations. Currently, we implement a specific aggregation protocol based on Paillier homomorphic encryption; see below for details and planned features.

```python
import tensorflow as tf
import tensorflow_federated as tff
from federated_aggregations import paillier

paillier_factory = paillier.local_paillier_executor_factory()
tff.framework.set_default_executor(paillier_factory)

x = [[i, i + 1] for i in range(5)]  # data from 5 clients
x_type = tff.TensorType(tf.int32, [2])

@tff.federated_computation(tff.FederatedType(x_type, tff.CLIENTS))
def secure_paillier_addition(x):
  return tff.federated_secure_sum(x, 32)

result = secure_paillier_addition(x)
print(result)
>>> [10 15]
```

# Installation
Installation process is currently in flux, since we depend on code that hasn't been published in a stable release; please check back soon for simpler instructions. If you're committed, see the instructions [below](#development).

# Features
## Federated computation
Currently, we simply add an implementation of the [`tff.federated_secure_sum`](https://www.tensorflow.org/federated/api_docs/python/tff/federated_secure_sum) to the default TFF execution stack. We do not yet rewrite any higher-level APIs for federated averaging yet, but these should be straightforward to implement.

## Protocols
Currently, we implement secure aggregation via the Paillier homomorphic encryption scheme. This protocol is well suited for federated averaging with highly-available clients, e.g. federated learning between organizations. Please see the [accompanying blog post](#TODO) for details. We outsource the Paillier aggregation to a "Paillier service" run separately from the Server role in traditional FL. This fits into the bulletin-board style of FL, where a service separate from the coordinator is responsible for aggregating models securely, and the Server (i.e. coordinator) periodically pulls & decrypts the latest model from that service. This achieves the specific functionality outlined in [this section](https://github.com/tf-encrypted/rfcs/tree/master/20190924-tensorflow-federated#specific-encrypted-executors) of the corresponding RFC.

## Secure Channels
TensorFlow Federated does not implement point-to-point communication directly between placements; it instead routes all communications through a computation "driver" (i.e. the host running the TFF Python script, also usually responsible for unplaced computation). To reduce communication at runtime, this driver is usually colocated with the tff.SERVER placement's executor stack, so that any values communicated between the driver and the tff.SERVER don't incur a network cost.

This communication pattern presents a problem for implementing secure aggregation, since many SMPC protocols assume the existence of authenticated channels between parties. In order to realize this in the specific case of a bulletin-board aggregation service, we follow the approach outlined in [this section](https://github.com/tf-encrypted/rfcs/tree/master/20190924-tensorflow-federated#network-strategy-and-secure-channels) of our RFC. Please see the [accompanying blog post](#TODO) for more details.

# Development
If you want to get up and running, please follow these steps. We strongly encourage using a virtual environment.
1. Install dependencies
    - Install `tf-big` from source, using the latest on `master` ([instructions](https://github.com/tf-encrypted/tf-big#development)).
    - Install `tfe-primitives` from source using Bazel ([instructions]())
    - Install TF Federated from source, using the most recent commit on [this branch](https://github.com/tf-encrypted/federated/tree/master). For guidance on installing TFF from source, you can follow the instructions provided [here](https://github.com/tensorflow/federated/blob/master/docs/install.md#build-the-tensorflow-federated-pip-package).
2. Install this package using pip (e.g. `pip install -e .`).
3. Run tests.

NOTE: tf-big & tfe-primitives currently only work with TF 2.1, while TFF uses tf-nightly to build it's pip package. We recommend building, installing, and testing tf-big and tfe-primitives with TF 2.1 _before_ installing the TF Federated dependency. If you happen to install tf-nightly while building the TFF pip package, you can force uninstall tf-nightly. Everything should still work with TF 2.1 instead.

If you run into issues, please [reach out](#support-and-feedback).

# Roadmap
Please see the original [TFF Integration RFC](https://github.com/tf-encrypted/rfcs/tree/master/20190924-tensorflow-federated) for an overview of project goals. While the implementation in this project has evolved, the high-level objectives have not changed.

# Support and Feedback
Bug reports and feature requests? Please [open an issue](https://github.com/tf-encrypted/federated-aggregations/issues) on Github.

For any other questions or feedback, please reach out directly on [Slack](https://join.slack.com/t/tf-encrypted/shared_invite/enQtNjI5NjY5NTc0NjczLWM4MTVjOGVmNGFkMWU2MGEzM2Q5ZWFjMTdmZjdmMTM2ZTU4YjJmNTVjYmE1NDAwMDIzMjllZjJjMWNiMTlmZTQ), or send an email to [contact@tf-encrypted.io](mailto:contact@tf-encrypted.io).

# License

Licensed under Apache License, Version 2.0 (see [LICENSE](./LICENSE) or http://www.apache.org/licenses/LICENSE-2.0). Copyright as specified in [NOTICE](./NOTICE).
