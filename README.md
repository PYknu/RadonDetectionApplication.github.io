# RadonDetectionApplication.github.io


    import os, sys
    from kivy.app import App
    import cv2
    import numpy as np
    import pandas as pd
    from kivy.config import Config
    from kivy.uix.textinput import TextInput
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.image import Image
    from kivy.clock import Clock
    from kivy.graphics.texture import Texture
    import statsmodels.api as sm
    from sklearn.preprocessing import StandardScaler

    Config.set('graphics','resizable', '0')
    Config.set('graphics','width', '1000')
    Config.set('graphics','height', '1000')

    class MyApp(App):
        def build(self):
            self.img1 = Image()
            self.layout = BoxLayout()
            self.layout.add_widget(self.img1)
            self.capture = cv2.VideoCapture(0)
            self.textinput = TextInput(text='Снизить концентрацию радона в уже существующих зданиях позволяет принятие следующих мер:\n1) более интенсивная вентиляция  подпольного пространства;\n2) обустройство системы отвода радона в основании здания или под монолитным полом на грунтовом основании;\n3) предотвращение поступления радона из подвального пространства в жилые помещения;\n4) заделка трещин и щелей в полах и стенах;\n5) улучшение вентилирования помещений.', multiline=True, font_size = 24)
            self.layout.add_widget(self.textinput)
            Clock.schedule_interval(self.update, 1.0 / 33.0)
            return self.layout

        def update(self,dt):
            _, imageFrame = self.capture.read()

            hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

            blue_lower = np.array([94, 80, 2], np.uint8)
            blue_upper = np.array([120, 255, 255], np.uint8)
            blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

            kernal = np.ones((5, 5), "uint8")

            blue_mask = cv2.dilate(blue_mask, kernal)
            res_blue = cv2.bitwise_and(imageFrame, imageFrame,
                                       mask=blue_mask)

            contours, hierarchy = cv2.findContours(blue_mask,
                                                   cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)

            input_file = "RadonData3.csv"
            df = pd.read_csv(input_file, header=0)
            scale = StandardScaler()
            X = df[['Number of particles', 'Humidity']]
            y = df['Alfarad concentration']

            X[['Number of particles', 'Humidity']] = scale.fit_transform(X[['Number of particles', 'Humidity']].values)
            est = sm.OLS(y, X).fit()
            est.summary()


            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if (area > 300):
                    x, y, w, h = cv2.boundingRect(contour)
                    imageFrame = cv2.rectangle(imageFrame, (x, y),
                                               (x + w, y + h),
                                             (255, 0, 0), 1)
                    number_of_objects_in_image = int(area) / 100
                    scaled = scale.transform([[number_of_objects_in_image, 25]])
                    predicted = est.predict(scaled[0])
                    cv2.putText(imageFrame, str(int(predicted)), (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (255, 0, 0))


            buf1 = cv2.flip(imageFrame, 0)
            buf = buf1.tostring()
            self.texture1 = Texture.create(size=(imageFrame.shape[1], imageFrame.shape[0]), colorfmt='bgr')
            self.texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img1.texture = self.texture1

    if __name__ == "__main__":
        MyApp().run()
        cv2.destroyAllWindows()
