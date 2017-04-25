import multiprocessing
import requests
import sys
import threading
from timeit import Timer


def request_item(item_id):
    print "thread before: {}".format(threading.currentThread().getName())
    try:
        r = requests.get("http://hn.algolia.com/api/v1/items/%s" % item_id)
        print "thread after: {}".format(threading.currentThread().getName())
        return r.json()
    except requests.RequestException:
        return None

def request_sequential():
    sys.stdout.write("Requesting sequentially...\n")

    results = []

    for i in range(1,21):

        results.append(request_item(i))

    print results


    sys.stdout.write("done.\n")

def request_concurrent():
    sys.stdout.write("Requesting in parallel...\n")
    items = range(1,21)
    thread_list = []
    for id_ in items:
        t = threading.Thread(target =request_item, args=(id_,))
        thread_list.append(t)
        t.start()
    for mythread in thread_list:
        mythread.join()


if __name__ == '__main__':
    t = Timer(lambda: request_sequential())
    print "Completed sequential in %s seconds." % t.timeit(1)
    print "--------------------------------------"

    t = Timer(lambda: request_concurrent())
    print "Completed using threads in %s seconds." % t.timeit(1)
