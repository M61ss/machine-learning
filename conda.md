# Basics

## Environment list

```shell
conda env list
```

## Create an environment

```shell
conda create -n <env_name>
```

## Activate environment

```shell
conda activate <env_name>
```

## Deactivate environment

```shell
conda deactivate
```

# Install packages into environments

## Using conda

```shell
conda install <package_name>
```

It is also possible to install packages from the outside of an environment:

```shell
conda install <package_name> --name <env_name>
```

## Using pip

```shell
conda install pip
```

Then, you can use pip to install packages in that environment:

```shell
pip install <package_name>
```

## Specify a version of a package

It works for both conda and pip:

- **Exact**: `qtconsole==4.5.1` means 4.5.1.
- **Fuzzy**: `qtconsole=4.5` means 4.5.0, 4.5.1, ..., etc.
- **\>=, >, <, <=**: `"qtconsole>=4.5"` means 4.5.0 or higher, `"qtconsole<4.6"` means less than 4.6.0.
- **OR**: `"qtconsole=4.5.1|4.5.2"` means 4.5.1, 4.5.2.
- **AND**: `"qtconsole>=4.3.1,<4.6"` means 4.3.1 or higher but less than 4.6.0.

## Update packages

```shell
conda update <package1_name> <package2_name> <...>
```

## Clone environment

```shell
conda create --name <clone_env_name> --clone <env_name>
```