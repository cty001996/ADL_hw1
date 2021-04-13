# Code for Homework 1 ADL NTU 109 Spring

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl"
make
# otherwise
pip install -r requirements.txt
```

## Preprocessing for train and test
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent detection training
```shell
python train_intent.py
```

## Slot tag training
```shell
python train_slot.py
```


## download model for test
```shell
bash download.sh
```

## Intent detection testing
```shell
bash intent_cls.sh
```

## Slot tag testing
```shell
bash slot_tag.sh
```
