import os
import time
try:
    while True:
        os.system('python manage.py runserver')
        os.system('rm -f core*')
        time.sleep(3)
except KeyboardInterrupt:
    print('stop!')