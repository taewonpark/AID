import os
import requests, io, tarfile
import shutil

if not os.path.exists("babi/data/en-valid-10k"):
    download_url = 'http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz'
    download_dir = 'tasks_1-20_v1-2'

    r = requests.get(download_url)
    t = tarfile.open(fileobj=io.BytesIO(r.content), mode='r|gz')
    t.extractall('.')

    shutil.move("tasks_1-20_v1-2/en-valid-10k", "babi/data/en-valid-10k")
    shutil.rmtree("tasks_1-20_v1-2/")