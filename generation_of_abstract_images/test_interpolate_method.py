from PIL import Image

if __name__ == '__main__':
   dict_methods = {'NEAREST': Image.NEAREST,
                   'BOX': Image.BOX,
                   'BILINEAR': Image.BILINEAR,
                   'LINEAR': Image.LINEAR,
                   'HAMMING': Image.HAMMING,
                   'BICUBIC': Image.BICUBIC,
                   'CUBIC': Image.CUBIC,
                   'LANCZOS': Image.LANCZOS,
                   'ANTIALIAS': Image.ANTIALIAS}
   image = Image.open('interpolate_test/image.png')
   new_width = image.size[0] * 15
   new_height = image.size[1] * 15
   for i in dict_methods.keys():
       resized_image = image.resize((new_width, new_height), dict_methods.get(i))
       resized_image.save(f"interpolate_test/image_{i}.png")