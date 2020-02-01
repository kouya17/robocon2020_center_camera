import logging
from websocket_server import WebsocketServer
import json
import threading
 
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(' %(module)s -  %(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

client_list = {}


# Callback functions
def new_client(client, server):
    logger.info('New client {}:{} has joined.'.format(client['address'][0], client['address'][1]))
    client_list[str(client['id'])] = {}
    client_list[str(client['id'])]['client'] = client


def client_left(client, server):
    logger.info('Client {}:{} has left.'.format(client['address'][0], client['address'][1]))


def message_received(client, server, message):
    logger.info('Message has been received from {}:{}'.format(client['address'][0], client['address'][1]))
    try:
        message_dict = json.loads(message)
        if message_dict['type'] == 'color':
            try:
                client_list[str(client['id'])]['color'] = [[message_dict.get('color', {}).get('H', {}).get('min', 0),
                                                            message_dict.get('color', {}).get('H', {}).get('max', 0)],
                                                           [message_dict.get('color', {}).get('S', {}).get('min', 0),
                                                            message_dict.get('color', {}).get('S', {}).get('max', 0)],
                                                           [message_dict.get('color', {}).get('V', {}).get('min', 0),
                                                            message_dict.get('color', {}).get('V', {}).get('max', 0)]]
            except AttributeError:
                logger.info('faild to get color data')
                client_list[str(client['id'])]['color'] = [[0, 0], [0, 0], [0, 0]]
    except json.JSONDecodeError:
        logger.info('faild to decode message to json')
    logger.info('client_list : {}'.format(client_list))


# Main
if __name__ == "__main__":
    server = WebsocketServer(port=9001, host='192.168.100.123', loglevel=logging.INFO)
    server.set_fn_new_client(new_client)
    server.set_fn_client_left(client_left)
    server.set_fn_message_received(message_received)
    
    th_server_run = threading.Thread(target=server.run_forever())
    th_server_run.start()
