using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Diagnostics;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.GPU;
using Emgu.CV.UI;

namespace facedetection1
{
    public partial class Form1 : Form
    {
        public List<Rectangle> kotak = new List<Rectangle>();
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            List<Rectangle> kotak = new List<Rectangle>();
            OpenFileDialog dlg = new OpenFileDialog();
            dlg.InitialDirectory = "C:\\";
            dlg.Filter = "video files (*.jpg; *.bmp; *.gif)|*.jpg; *.bmp; *.gif|All Files (*.*)|*.*";
            dlg.RestoreDirectory = true;

            if (dlg.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                string selectedFileName = dlg.FileName;
                Image<Bgr,byte> gambarx=new Image<Bgr,byte>(selectedFileName);
                Image<Bgr,byte> imagex = detection(gambarx, kotak);
                imageBox1.Image = imagex;
                Bitmap wajah=imagex.ToBitmap();
                imageBox2.Image = new Image<Bgr, byte>(wajah.Clone(kotak[0], wajah.PixelFormat));
                
            }
        }

        private static Image<Bgr,byte> detection (Image<Bgr,byte> image, List<Rectangle> kotak)
        {
            //Image<Bgr, Byte> image = new Image<Bgr, byte>("lena.jpg"); //Read the files as an 8-bit Bgr image  
            
            Stopwatch watch;
            String faceFileName = "haarcascade_frontalface_default.xml";
            String eyeFileName = "haarcascade_eye.xml";

            if (GpuInvoke.HasCuda)
            {
                using (GpuCascadeClassifier face = new GpuCascadeClassifier(faceFileName))
                using (GpuCascadeClassifier eye = new GpuCascadeClassifier(eyeFileName))
                {
                    watch = Stopwatch.StartNew();
                    using (GpuImage<Bgr, Byte> gpuImage = new GpuImage<Bgr, byte>(image))
                    using (GpuImage<Gray, Byte> gpuGray = gpuImage.Convert<Gray, Byte>())
                    {
                        Rectangle[] faceRegion = face.DetectMultiScale(gpuGray, 1.1, 10, Size.Empty);
                        foreach (Rectangle f in faceRegion)
                        {
                            //draw the face detected in the 0th (gray) channel with blue color
                            image.Draw(f, new Bgr(Color.Blue), 2);
                            using (GpuImage<Gray, Byte> faceImg = gpuGray.GetSubRect(f))
                            {
                                //For some reason a clone is required.
                                //Might be a bug of GpuCascadeClassifier in opencv
                                using (GpuImage<Gray, Byte> clone = faceImg.Clone())
                                {
                                    Rectangle[] eyeRegion = eye.DetectMultiScale(clone, 1.1, 10, Size.Empty);

                                    foreach (Rectangle e in eyeRegion)
                                    {
                                        Rectangle eyeRect = e;
                                        eyeRect.Offset(f.X, f.Y);
                                        image.Draw(eyeRect, new Bgr(Color.Red), 2);
                                    }
                                }
                            }
                        }
                    }
                    watch.Stop();
                }
            }
            else
            {
                //Read the HaarCascade objects
                List<Bitmap> gambar = new List<Bitmap>();
                using (HaarCascade face = new HaarCascade(faceFileName))
                using (HaarCascade eye = new HaarCascade(eyeFileName))
                {
                    watch = Stopwatch.StartNew();
                    using (Image<Gray, Byte> gray = image.Convert<Gray, Byte>()) //Convert it to Grayscale
                    {
                        //normalizes brightness and increases contrast of the image
                        gray._EqualizeHist();

                        //Detect the faces  from the gray scale image and store the locations as rectangle
                        //The first dimensional is the channel
                        //The second dimension is the index of the rectangle in the specific channel
                        MCvAvgComp[] facesDetected = face.Detect(
                           gray,
                           1.1,
                           10,
                           Emgu.CV.CvEnum.HAAR_DETECTION_TYPE.DO_CANNY_PRUNING,
                           new Size(20, 20));
                        
                        foreach (MCvAvgComp f in facesDetected)
                        {
                            //draw the face detected in the 0th (gray) channel with blue color
                            image.Draw(f.rect, new Bgr(Color.Blue), 2);
                            kotak.Add(f.rect);
                            //Set the region of interest on the faces
                            gray.ROI = f.rect;
                            MCvAvgComp[] eyesDetected = eye.Detect(
                               gray,
                               1.1,
                               10,
                               Emgu.CV.CvEnum.HAAR_DETECTION_TYPE.DO_CANNY_PRUNING,
                               new Size(20, 20));
                            gray.ROI = Rectangle.Empty;

                            foreach (MCvAvgComp e in eyesDetected)
                            {
                                Rectangle eyeRect = e.rect;
                                eyeRect.Offset(f.rect.X, f.rect.Y);
                                image.Draw(eyeRect, new Bgr(Color.Red), 2);
                            }
                        }
                    }
                    watch.Stop();
                }
            }

            //display the image 
            //ImageViewer.Show(image, String.Format(
              // "Completed face and eye detection using {0} in {1} milliseconds",
               //GpuInvoke.HasCuda ? "GPU" : "CPU",
               //watch.ElapsedMilliseconds));
            return image;
        }
    }
}
