# Instructions for using Bath Hex
## login to ssh
ssh to one of the msc servers - https://hex.cs.bath.ac.uk/usage
For example:

```
ssh rwg27@ogg.cs.bath.ac.uk
```

The password is your normal uni password for engage etc

## set up ssh key

```
ssh-keygen -t rsa -b 4096 -C "YOUR GITHUB EMAIL ADDRESS HERE"
cat ~/.ssh/id_rsa.pub
```

copy to github -> settings -> ssh keys in github

## Validate github access

```
ssh -T git@github.com
```

## Clone

create dir and checkout repo
```
git clone git@github.com:kayfadeyi/montezuma-revenge-exploration-rl.git
```

## pull latest code

```
git pull origin main
```

## disk space

```
df -h
du -sh ~
```

## build container

hare build -t <my username>/<my image name> .
e.g.
```
hare build -t rwg27/rl2 .
```

## run container
run the container - remove after running
```
hare run --rm -v "$(pwd)":/app --user $(id -u):$(id -g) rwg27/rl2
```
run the container - Detached mode - no remove
```
hare run -d --name my_rl2_container -v "$(pwd)":/app --user $(id -u):$(id -g) rwg27/rl2
```
run the container - Detached mode - no remove - with gpus
```
hare run -d --name my_rl2_container -v "$(pwd)":/app --user $(id -u):$(id -g) --gpus '"device=1"' rwg27/rl2
```
reattach to the container
```
hare attach my_rl2_container
```

# stop the container
```
hare stop my_rl2_container
```
# remove the container
```
hare rm my_rl2_container
```

# logs - follow
```
hare logs -f my_rl2_container
```
```
hare logs my_rl2_container
```


