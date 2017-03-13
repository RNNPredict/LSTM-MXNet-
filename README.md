# 複数レイヤ LSTM ネットワーク（MXNet 実装）

## チュートリアル

まずは手許の環境でどんな学習・推論ができるのかを試してみましょう。  

### 1. docker-compose のインストール

以下のリンクから、Docker をインストールしてください。

- [Mac](https://docs.docker.com/docker-for-mac/install/#download-docker-for-mac)
- [Windows](https://docs.docker.com/docker-for-windows/install/)

ターミナルで、docker-compose も同時にインストールされたことを確認します。

```
$ docker-compose -v

docker-compose version 1.11.2, build dfed245
```

### 2. docker-compose で Jupyter を起動

以下のコマンドで Jupyter を起動します

```
$ docker-compose up

Creating network "mxnetcharlstm_default" with the default driver
Creating jupyter
Attaching to jupyter
jupyter    | [I 20:36:10.489 NotebookApp] Writing notebook server cookie secret to /root/.local/share/jupyter/runtime/notebook_cookie_secret
jupyter    | [I 20:36:10.507 NotebookApp] Serving notebooks from local directory: /root/notebook
jupyter    | [I 20:36:10.507 NotebookApp] 0 active kernels
jupyter    | [I 20:36:10.509 NotebookApp] The Jupyter Notebook is running at: http://0.0.0.0:8888/?token=ac29066ff4b4e131ea28317aed8b63069ba9a5ae410e2d18
jupyter    | [I 20:36:10.509 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
jupyter    | [C 20:36:10.510 NotebookApp]
jupyter    |
jupyter    |     Copy/paste this URL into your browser when you connect for the first time,
jupyter    |     to login with a token:
jupyter    |         http://0.0.0.0:8888/?token=ac29076ff4b4e131ea28317aed8b63069ba9a5ae410e2d17
```

### 3. ブラウザで Juputer を開きます

[http://localhost:8888](http://localhost:8888)

トークンを要求されるので、ターミナルに表示されているトークンを入力します。  
`tutorials.ipynb` をクリックして開き、チュートリアルを開始します。

### 4. 環境の破棄

チュートリアルが終わったら、ターミナルにもどり `Ctrl + C` でログ監視から抜け  
以下のコマンドで完全に Jupyter を停止しましょう。

```
$ docker-compose down -v

Stopping jupyter ... done
Removing jupyter ... done
Removing network mxnetcharlstm_default
```
