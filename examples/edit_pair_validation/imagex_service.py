from bytedance.vcloud.imagex.ImageXService import ImageXService
from retrying import retry
from PIL import Image
import io
import os
from typing import Union, List


class ImgXService():
    def __init__(self):
        self.cli = ImageXService('ap-singapore-1')
        self.ak = os.environ['VOLCENGINE_ACCESS_KEY_ID']
        self.sk = os.environ['VOLCENGINE_SECRET_ACCESS_KEY']
        self.service_id = os.environ.get('IMAGEX_SERVICE_ID', '375lmtcpo0')

        self.cli.set_ak(self.ak)
        self.cli.set_sk(self.sk)

    @retry(stop_max_attempt_number=3)
    def imagex_upload(self, imgs: Union[List[str], List[bytes], List[os.PathLike]], params_input = None):
        # try:
        params = dict()
        if params_input is not None:
            params.update(params_input)
        params['ServiceId'] = self.service_id
        # ApplyImageUpload params
        # params['SessionKey'] = 'pangle_aigc_banner'
            # ['StoreKeys'] = store_keys
        # params['FileExtension'] = ''

        # CommitImageUpload params
        # params['SkipMeta'] = False
        # params['Functions'] = []

        # False: 通过TOS上传(default)；
        # True： 通过下发域名进行上传
        params['UploadByHost'] = True
        # 跳过commit阶段
        params['SkipCommit'] = False

        # 图片上传携带身份
        # params['uid'] = '114514'
        # params['appid'] = '1128'
        if isinstance(imgs[0],str):
            # 通过选取本地文件路径进行上传
            resp = self.cli.upload_img_files(params, imgs)
        elif isinstance(imgs[0],Image.Image):
            datas = [self.image2iobuffer(file) for file in imgs]
            resp = self.cli.upload_images(params, datas)
        elif isinstance(imgs[0], bytes):
            resp = self.cli.upload_images(params, imgs)
        else:
            return []
        urls = []
        for idx,r in enumerate(resp['PluginResult']):
            assert resp["Results"][idx]["UriStatus"] == 2000, "upload image %d falied, uri %s, status code %s\n" % (idx,r["ImageUri"], resp["Results"][idx]["UriStatus"])
            urls.append('https://p16-lp-sg.ibyteimg.com/{}~tplv-375lmtcpo0-image.{}'.format(r['ImageUri'], r['ImageFormat'] if r['ImageFormat'] != '' else 'png'))

        return urls
    
    def image2iobuffer(self, image:Image.Image, format="PNG"):
        # Create an in-memory buffer to store the PNG image data
        png_image_buffer = io.BytesIO()

        # Save the PIL Image object as PNG to the buffer
        image.save(png_image_buffer, format=format, lossless=True)

        # Get the PNG image data from the buffer
        png_image_data = png_image_buffer.getvalue()

        return png_image_data


if __name__ == '__main__':
    imgx_ser = ImgXService()
    print(imgx_ser.imagex_upload(imgs = ['/cloudide/workspace/aigc_image_service/matx_image.png']))

        
