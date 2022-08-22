### Create Virtual Environment venv

Python must be installed

#### Linux

```
bash virtualenv venv
```

#### Windows

```bash
python -m venv ./venv
```

### Activate Environment
maybe the windows-secuirty has problems because of used scripts:

```bash
Set-ExecutionPolicy Unrestricted -Scope Process
```
the activation of venv should work now only in the chosen terminal

#### Windows

```bash
.\venv\Scripts\activate
```

#### Linux

```bash
source venv/Scripts/activate
```

#### MacOS

```bash
source venv/bin/activate
```

### Install Requirements

```bash
pip install -r requirements.txt
```

### Install RLCard

```bash
pip3 install -e .
```

#### Rreeze Requirements

```bash
pip freeze > requirements.txt
```

### Deactivate Env

#### Windows

```bash
deactivate
```

#### Linux

```bash
source deactivate
```