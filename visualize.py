from PIL import Image, ImageDraw
import argparse

from dataset import read_label_file, convert_pcd

parser = argparse.ArgumentParser()
parser.add_argument('partial_path')

args = parser.parse_args()

green = (0, 255, 0)
red = (255, 0, 0)

def draw_box(p1, p2, p3, p4, draw, color):
    ps = [p1, p2, p3, p4]
    for i in xrange(4):
        x1, y1 = ps[i]
        x2, y2 = ps[(i+1) % 4]
        if i % 2 == 1:
            draw.line((x1, y1, x2, y2), fill=color)
        else:
            draw.line((x1, y1, x2, y2))

with Image.open('%sr.png' % args.partial_path) as img:
    draw = ImageDraw.Draw(img)
    for box in read_label_file('%scpos.txt' % args.partial_path):
        draw_box(box[0], box[1], box[2], box[3], draw, green)

    for box in read_label_file('%scneg.txt' % args.partial_path):
        draw_box(box[0], box[1], box[2], box[3], draw, red)

    img.show()

depth_data = convert_pcd('%s.txt' % args.partial_path)
depth_img = Image.new('F', (640, 480))
# strange scale in the data
depth_img.putdata(depth_data.flatten() * 1e40)
depth_img.show()
