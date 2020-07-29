## About

This repository contains some boilerplate code for running simple
machine learning simulations that are intended to be reproduced for some
purpose. Functionality is provided for easily decoupling the code concerning data,
models and training algorithms, tracking custom made metrics throughout the
training process, and finally, running "embarassingly parallel" simulations in
a reproducible manner on the specified cpus or gpus. The code is built
on top of [ignite](https://pytorch.org/ignite/), which is still actively
developed; hence, some of the functionality introduced in this repository
might be redundant (or will become so in the future).

I share this code between my projects (e.g., see
[early-stopped-mirror-descent](https://github.com/TomasVaskevicius/early-stopped-mirror-descent)).
Backward compatibility is not intended to be maintained. Anyone that finds the code
contained in this repository of any use is welcome to use it without crediting me.

## Dependencies

* [pytorch](https://pytorch.org/)
* [torchvision](https://pytorch.org/)
* [ignite](https://pytorch.org/ignite/)

