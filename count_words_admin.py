from rq import Queue
from redis import Redis
from count_urls import count_words_at_url
import time

# Tell RQ what Redis connection to use
redis_conn = Redis(host='192.168.1.127')
q = Queue(connection=redis_conn)  # no args implies the default queue

# Delay execution of count_words_at_url('http://nvie.com')
job = q.enqueue(count_words_at_url, 'http://nvie.com')
print(job.result)   # => None

# Now, wait a while, until the worker is finished
while True:
    time.sleep(2)
    print(job.result)   # => 889