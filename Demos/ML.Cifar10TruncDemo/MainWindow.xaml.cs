using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace PicturePrimitive
{
  public struct RGB
  {
    public RGB(byte r, byte g, byte b)
    {
      R=r;
      G=g;
      B=b;
    }

    public byte R;
    public byte G;
    public byte B;

    public System.Windows.Media.Color ToColor()
    {
      return System.Windows.Media.Color.FromRgb(R, G, B);
    }

    public bool Equals(RGB other)
    {
      return (R == other.R) && (G == other.G) && (B == other.B);
    }

    public override bool Equals(object other)
    {
      if (!(other is RGB)) return false;
      return this.Equals((RGB)other);
    }

    public override int GetHashCode()
    {
      return R ^ G ^ B;
    }
  }

  /// <summary>
  /// Interaction logic for MainWindow.xaml
  /// </summary>
  public partial class MainWindow : Window
  {
    #region CONST

    public const int NORM_IMG_SIZE = 32;

    #endregion

    public MainWindow()
    {
      InitializeComponent();
    }

    private BitmapImage m_Bitmap;

    #region Load Image

    private void onAddButtonClick(object sender, RoutedEventArgs e)
    {
      var dialog = new Microsoft.Win32.OpenFileDialog();
      dialog.Filter = "Image Files (*.jpg; *.jpeg; *.gif; *.bmp)|*.jpg; *.jpeg; *.gif; *.bmp";

      if ((bool)dialog.ShowDialog())
      {
        loadImage(dialog.FileName);
      }
    }

    private void onImageDrop(object sender, DragEventArgs e)
    {
      if (e.Data.GetDataPresent(DataFormats.FileDrop))
      {
        var files = (string[])e.Data.GetData(DataFormats.FileDrop);
        if (!files.Any()) return;

        loadImage(files[0]);
      }
    }

    private void loadImage(string path)
    {
      m_DropHereTxt.Visibility = Visibility.Collapsed;
      m_ImgInitial.Visibility = Visibility.Visible;
      m_Bitmap = new BitmapImage(new Uri(path));
      m_ImgInitial.Source = m_Bitmap;

      doPrimitivization();
    }

    #endregion

    #region Primitivization

    private void doPrimitivization()
    {
      var result = doNormalization();



    }

    private RGB[,] doNormalization()
    {
      var xm = m_Bitmap.PixelWidth;
      var ym = m_Bitmap.PixelHeight;
      var cm = Math.Min(xm, ym);

      //var crop = new CroppedBitmap(m_Bitmap, new Int32Rect((xm-cm)/2, (ym-cm)/2, cm, cm));
      //cro
      //
      //
      //using (var gr = Graphics.FromImage(crop))
      //{
      //  gr.SmoothingMode      = SmoothingMode.HighQuality;
      //  gr.InterpolationMode  = InterpolationMode.HighQualityBicubic;
      //  gr.PixelOffsetMode    = PixelOffsetMode.HighQuality;
      //  gr.CompositingQuality = CompositingQuality.HighQuality;
      //  gr.DrawImage(m_Bitmap, new System.Drawing.Rectangle((xm-cm)/2, (ym-cm)/2, cm, cm));
      //}

      // normalize image to NORM_IMG_SIZE x NORM_IMG_SIZE

      var result = new RGB[NORM_IMG_SIZE, NORM_IMG_SIZE];


      //var lambda = (cm-1.0D)/(NORM_IMG_SIZE-1.0D);
      //
      //var stride = xm*4;
      //var size = ym*stride;
      //var pixels = new byte[size];
      //m_Bitmap.CopyPixels(pixels, stride, 0);
      //
      //for (int i=0; i<NORM_IMG_SIZE; i++)
      //{
      //  var ymin = lambda*i + (ym-1-lambda*(NORM_IMG_SIZE-1))/2;
      //  var ymax = ymin+lambda;
      //
      //  for (int j=0; j<NORM_IMG_SIZE; j++)
      //  {
      //    var rsumm = 0.0D;
      //    var gsumm = 0.0D;
      //    var bsumm = 0.0D;
      //
      //    var xmin = lambda*j + (xm-1-lambda*(NORM_IMG_SIZE-1))/2;
      //    var xmax = xmin+lambda;
      //
      //    int total = 0;
      //    for (int y=(int)ymin; y<ymax; y++)
      //    for (int x=(int)xmin; x<xmax; x++)
      //    {
      //      if (x<0 || x>=xm || y<0 || y>=ym) continue;
      //
      //      int index = y*stride + 4*x;
      //      bsumm += pixels[index];
      //      gsumm += pixels[index + 1];
      //      rsumm += pixels[index + 2];
      //      total++;
      //    }
      //
      //    result[i, j] = total > 0 ?
      //                   new RGB((byte)(rsumm / total), (byte)(gsumm / total), (byte)(bsumm / total)) :
      //                   new RGB(255, 255, 255);
      //  }
      //}
      //
      //// draw result
      //m_NormalizedCanvas.Children.Clear();
      //for (var y=0; y<NORM_IMG_SIZE; y++)
      //for (var x=0; x<NORM_IMG_SIZE; x++)
      //{
      //  var rgb = result[y, x];
      //  var pixel = new Rectangle
      //  {
      //    Margin=new Thickness(0),
      //    Height=5, Width=5,
      //    Fill=new SolidColorBrush(Color.FromRgb(rgb.R, rgb.G, rgb.B))
      //  };
      //  m_NormalizedCanvas.Children.Add(pixel);
      //  Canvas.SetLeft(pixel, x*5);
      //  Canvas.SetTop(pixel, y*5);
      //}

      return result;
    }


    #endregion
  }
}
