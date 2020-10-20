import pygame
import numpy as np
from PIL import Image, ImageDraw


class Visualiser(object):

    def __init__(self, width=1000, height=500, sidebar_width=200, sidebar_pad=5, mode="top"):
        self.width = width
        self.height = height
        self.sidebar_width = sidebar_width
        self.sidebar_pad = sidebar_pad
        self.thumb_width = sidebar_width
        self.thumb_height = sidebar_width / 2
        self.screen = None
        self.__done = True
        self.__mode = mode
        self.__caption = ""
        self.__sub_caption = ""

    @property
    def mode(self):
        return self.__mode

    def reset(self, mode=None):
        pygame.init()
        if mode is not None:
            self.set_mode(mode)

        if self.__mode == "top":
            self.screen = pygame.display.set_mode((self.width + self.sidebar_width + self.sidebar_pad, self.width))
        else:
            self.screen = pygame.display.set_mode((self.width, self.height))

        self.__done = False

    def set_mode(self, new_mode):
        self.__mode = new_mode

    def update_main(self, img_func, en=None, thumbs=None, caption=""):
        if self.__mode == "top":
            img = img_func(width=self.width, length=self.width)
        else:
            img = img_func(width=self.width, height=self.height)
        self.screen.blit(pygame.image.fromstring(img.tobytes("raw", "RGB"), img.size, "RGB"), (0, 0))

        if en is not None:
            en_img = self.__draw_en(en)
            self.screen.blit(
                pygame.image.fromstring(en_img.tobytes("raw", "RGB"), en_img.size, "RGB"),
                (self.width + self.sidebar_pad, self.width - self.thumb_height)
            )
        if thumbs is not None:
            if en is not None:
                thumb = thumbs[np.argmin(np.array(en))]
            else:
                thumb = thumbs if type(thumbs) is Image.Image else thumbs[0]
            thumb = thumb.resize((200, 100))
            self.screen.blit(
                pygame.image.fromstring(thumb.tobytes("raw", "RGB"), thumb.size, "RGB"),
                (self.width + self.sidebar_pad, 0)
            )

        pygame.display.flip()
        if caption is not None:
            self.__caption = caption
            self.update_caption()

    def update_thumb(self, img, pn=None, pn_mode="RGB", caption=None):
        """

        :param img:
        :type img: Image.Image
        :param pn:
        :type pn: np.ndarray
        :param pn_mode:
        :type pn_mode: basestring
        :param caption:
        :type caption: basestring
        :return:
        """
        if self.__mode == "panorama":
            return

        pn_img = img.resize((self.thumb_width, self.thumb_height))
        self.screen.blit(
            pygame.image.fromstring(pn_img.tobytes("raw", "RGB"), pn_img.size, "RGB"),
            (self.width + self.sidebar_pad, 1 * (self.thumb_height + self.sidebar_pad))
        )
        pn_red, pn_grn, pn_blu = pn_img.split()
        self.screen.blit(
            pygame.image.fromstring(pn_red.convert("RGB").tobytes("raw", "RGB"), pn_red.size, "RGB"),
            (self.width + self.sidebar_pad, 2 * (self.thumb_height + self.sidebar_pad))
        )
        self.screen.blit(
            pygame.image.fromstring(pn_grn.convert("RGB").tobytes("raw", "RGB"), pn_grn.size, "RGB"),
            (self.width + self.sidebar_pad, 3 * (self.thumb_height + self.sidebar_pad))
        )
        self.screen.blit(
            pygame.image.fromstring(pn_blu.convert("RGB").tobytes("raw", "RGB"), pn_blu.size, "RGB"),
            (self.width + self.sidebar_pad, 4 * (self.thumb_height + self.sidebar_pad))
        )
        if pn is not None:
            pn_act = Image.new("RGB", (10, 36))

            if pn.size == 360:
                pn_act = Image.fromarray(np.int8(pn * 255).reshape((10, 36)), "L").convert("RGB")
            elif pn.size == 360 * 3:
                pn_act = Image.fromarray(np.int8(pn * 255).reshape((10, 36, 3)), "RGB")
            pn_act = pn_act.resize((self.thumb_width, self.thumb_height))

            if pn_mode == "RGB":
                self.screen.blit(
                    pygame.image.fromstring(pn_act.tobytes("raw", "RGB"), pn_act.size, "RGB"),
                    (self.width + self.sidebar_pad, 5 * (self.thumb_height + self.sidebar_pad))
                )
            else:
                pn_red, pn_grn, pn_blu = pn_act.split()
                self.screen.blit(
                    pygame.image.fromstring(pn_red.convert("RGB").tobytes("raw", "RGB"), pn_red.size, "RGB"),
                    (self.width + self.sidebar_pad, 5 * (self.thumb_height + self.sidebar_pad))
                )
                self.screen.blit(
                    pygame.image.fromstring(pn_grn.convert("RGB").tobytes("raw", "RGB"), pn_grn.size, "RGB"),
                    (self.width + self.sidebar_pad, 6 * (self.thumb_height + self.sidebar_pad))
                )
                self.screen.blit(
                    pygame.image.fromstring(pn_blu.convert("RGB").tobytes("raw", "RGB"), pn_blu.size, "RGB"),
                    (self.width + self.sidebar_pad, 7 * (self.thumb_height + self.sidebar_pad))
                )

        pygame.display.flip()
        if caption is not None:
            self.__sub_caption = caption
            self.update_caption()

    def update_caption(self):
        caption = self.__caption
        if not (self.__sub_caption == ""):
            caption += " | " + self.__sub_caption
        pygame.display.set_caption(caption)

    def is_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.__done = True
                return True
        return self.__done

    def __draw_en(self, en):
        """
        :param en:
        :type en: np.ndarray
        :return:
        """

        def en_trans(x):
            """

            :param x:
            :type x: np.ndarray
            :return:
            """
            return np.clip(x / 20., 0., 1.)

        w, h = self.thumb_width, self.thumb_height
        image = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.line(tuple(
            (w * i / 61., h * (1. - en_trans(e))) for i, e in enumerate(en)
        ), fill=(250, 0, 0))
        return image
