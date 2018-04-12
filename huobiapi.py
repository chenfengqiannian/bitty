from huobi_client import StreamingClient


def on_message(data):
    print(data)


sclient = StreamingClient()
sclient.subscribe_all()
sclient.connect(on_message)