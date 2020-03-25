import io as io
import PIL as pil
import urllib as ul


def get_url(url):
    opener = ul.request.build_opener()

    opener.addheaders = [
        (
            'User-Agent',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36'
        )]

    ul.request.install_opener(opener)
    response = ul.request.urlopen(url)

    return response.read()


def download_bytes(url):
    return io.BytesIO(
        get_url(url))


def download_image(url):
    return pil.Image.open(
        download_bytes(
            url))