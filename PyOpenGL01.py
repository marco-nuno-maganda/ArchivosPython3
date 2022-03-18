"""
https://rdmilligan.wordpress.com/2015/09/10/augmented-reality-using-opencv-and-opengl/
Modificaciones Realizadas en 2021, porque algunas libreras quedaron obsoletas
"""

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import cv2
import pygame
from PIL import Image
#from webcam import Webcam
#from detection import Detection
  
class HandTracker:
    def cv2ImageToSurface(self,cv2Image):
        if cv2Image.dtype.name == 'uint16':
            cv2Image = (cv2Image / 256).astype('uint8')
        size = cv2Image.shape[1::-1]
        if len(cv2Image.shape) == 2:
            cv2Image = np.repeat(cv2Image.reshape(size[1], size[0], 1), 3, axis = 2)
            format = 'RGB'
        else:
            format = 'RGBA' if cv2Image.shape[2] == 4 else 'RGB'
            cv2Image[:, :, [0, 2]] = cv2Image[:, :, [2, 0]]
        surface = pygame.image.frombuffer(cv2Image.flatten(), size, format)
        if format == 'RGBA':
            return (surface.convert_alpha())
        else:
            return (surface.convert())


    def load_texture_from_Webcam(self):
        tex_id = glGenTextures(1)
        #tex = pygame.image.load(texture_url)
        tex=self.cv2ImageToSurface(self.image)        
        tex_surface = pygame.image.tostring(tex, 'RGBA')
        
        tex_width, tex_height = tex.get_size()
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex_width, tex_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_surface)
        #glBindTexture(GL_TEXTURE_2D, 0)
        return tex_id
    

    def load_texture_from_file(self,texture_url):
        tex_id = glGenTextures(1)
        tex = pygame.image.load(texture_url)
        #tex=self.cv2ImageToSurface(self.image)        
        tex_surface = pygame.image.tostring(tex, 'RGBA')
        
        tex_width, tex_height = tex.get_size()
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex_width, tex_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_surface)
        return (tex_id)
        #glBindTexture(GL_TEXTURE_2D, 0)
        #return "x"


    def __init__(self):
    
        pygame.init()
        #screen = pygame.display.set_mode((640,480))
        flags = pygame.HIDDEN
        screen = pygame.display.set_mode((640,480), flags)        
    
        self.cap = cv2.VideoCapture(0)
  
        self.x_axis = 0.0
        self.z_axis = 0.0
        self.show_cube = True
        self.texture_background = None
        self.texture_cube = None
        
    def _init_gl(self, Width, Height):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(Width)/float(Height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
             
        # enable texture
        glEnable(GL_TEXTURE_2D)
        self.texture_background = glGenTextures(1)        
        #self.texture_cube = self.load_texture_from_file("board.png")
        self.texture_cube = self.load_texture_from_file("LogoUPV.png")
        
        self.qrCodeDetector = cv2.QRCodeDetector()
        
     

    def _draw_scene(self):
        # handle any hand gesture
        self._handle_gesture()
      
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        gluLookAt(0,0,10,0,0,0,0,1,0);
     
        # draw background
        glBindTexture(GL_TEXTURE_2D, self.idtextura)
        glPushMatrix()
        glTranslatef(0.0,0.0,-3.0)
        self._draw_background()
        glPopMatrix()
     
        # draw cube if enabled
        if self.show_cube:
            glColor4f(1.0, 1.0, 1.0, 1.0)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)
            #glEnable(GL_BLEND)
            #glDisable(GL_DEPTH_TEST)
     
            glBindTexture(GL_TEXTURE_2D, self.texture_cube)
            glPushMatrix()
            glTranslatef(0.0,0.0,-1.0)
            glRotatef(self.x_axis,1.0,0.0,0.0)
            glRotatef(0.0,0.0,1.0,0.0)
            glRotatef(self.z_axis,0.0,0.0,1.0)
            self._draw_cube()
            glPopMatrix()
     
            #glDisable(GL_BLEND)
            #glEnable(GL_DEPTH_TEST)
     
            # update rotation values
            self.x_axis = self.x_axis - 10
            self.z_axis = self.z_axis - 10
      
        glutSwapBuffers()

    def _handle_gesture(self):
        self.ret, self.image = self.cap.read()
        self.idtextura=self.load_texture_from_Webcam()
        
        data, bbox, _ = self.qrCodeDetector.detectAndDecode(self.image)
        # check if there is a QRCode in the image
        if bbox is not None:
            if data:
                print (bbox)
                print (bbox[0])
                print (bbox[0][0])
                print (bbox[0][1])
                print (bbox[0][2])
                print (bbox[0][3])
                #print (bbox[i+1][0])
                img = cv2.line(self.image,(int(bbox[0][0][0]),int(bbox[0][0][1])),(int(bbox[0][1][0]),int(bbox[0][1][1])),(0,0,255),3)
                img = cv2.line(self.image,(int(bbox[0][1][0]),int(bbox[0][1][1])),(int(bbox[0][2][0]),int(bbox[0][2][1])),(0,0,255),3)
                img = cv2.line(self.image,(int(bbox[0][2][0]),int(bbox[0][2][1])),(int(bbox[0][3][0]),int(bbox[0][3][1])),(0,0,255),3)
                img = cv2.line(self.image,(int(bbox[0][3][0]),int(bbox[0][3][1])),(int(bbox[0][0][0]),int(bbox[0][0][1])),(0,0,255),3)

                print("[+] QR Code detected, data:", data)
            
        
            
    def _draw_background(self):
        # draw background
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex3f(-8.0, -6.0, 0.0)
        glTexCoord2f(1.0, 1.0); glVertex3f( 8.0, -6.0, 0.0)
        glTexCoord2f(1.0, 0.0); glVertex3f( 8.0,  6.0, 0.0)
        glTexCoord2f(0.0, 0.0); glVertex3f(-8.0,  6.0, 0.0)
        glEnd( )
     
    def _draw_cube(self):
        Factor=0.5
        # draw cube
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0); glVertex3f(-Factor, -Factor,  Factor)
        glTexCoord2f(1.0, 0.0); glVertex3f( Factor, -Factor,  Factor)
        glTexCoord2f(1.0, 1.0); glVertex3f( Factor,  Factor,  Factor)
        glTexCoord2f(0.0, 1.0); glVertex3f(-Factor,  Factor,  Factor)
        glTexCoord2f(1.0, 0.0); glVertex3f(-Factor, -Factor, -Factor)
        glTexCoord2f(1.0, 1.0); glVertex3f(-Factor,  Factor, -Factor)
        glTexCoord2f(0.0, 1.0); glVertex3f( Factor,  Factor, -Factor)
        glTexCoord2f(0.0, 0.0); glVertex3f( Factor, -Factor, -Factor)
        glTexCoord2f(0.0, 1.0); glVertex3f(-Factor,  Factor, -Factor)
        glTexCoord2f(0.0, 0.0); glVertex3f(-Factor,  Factor,  Factor)
        glTexCoord2f(1.0, 0.0); glVertex3f( Factor,  Factor,  Factor)
        glTexCoord2f(1.0, 1.0); glVertex3f( Factor,  Factor, -Factor)
        glTexCoord2f(1.0, 1.0); glVertex3f(-Factor, -Factor, -Factor)
        glTexCoord2f(0.0, 1.0); glVertex3f( Factor, -Factor, -Factor)
        glTexCoord2f(0.0, 0.0); glVertex3f( Factor, -Factor,  Factor)
        glTexCoord2f(1.0, 0.0); glVertex3f(-Factor, -Factor,  Factor)
        glTexCoord2f(1.0, 0.0); glVertex3f( Factor, -Factor, -Factor)
        glTexCoord2f(1.0, 1.0); glVertex3f( Factor,  Factor, -Factor)
        glTexCoord2f(0.0, 1.0); glVertex3f( Factor,  Factor,  Factor)
        glTexCoord2f(0.0, 0.0); glVertex3f( Factor, -Factor,  Factor)
        glTexCoord2f(0.0, 0.0); glVertex3f(-Factor, -Factor, -Factor)
        glTexCoord2f(1.0, 0.0); glVertex3f(-Factor, -Factor,  Factor)
        glTexCoord2f(1.0, 1.0); glVertex3f(-Factor,  Factor,  Factor)
        glTexCoord2f(0.0, 1.0); glVertex3f(-Factor,  Factor, -Factor)
        glEnd()        
        
    def main(self):
        # setup and run OpenGL
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(640, 480)
        glutInitWindowPosition(800, 400)
        glutCreateWindow("OpenGL Hand Tracker")
        glutDisplayFunc(self._draw_scene)
        glutIdleFunc(self._draw_scene)
        self._init_gl(640, 480)
        glutMainLoop()
  
# run instance of Hand Tracker 
handTracker = HandTracker()
handTracker.main()        
