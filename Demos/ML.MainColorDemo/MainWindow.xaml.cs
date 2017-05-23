using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.IO;
using System.Net;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;

using ML.Core;
using ML.DeepMethods.Models;
using System.Drawing.Imaging;

namespace ML.CatDogDemo
{
  /// <summary>
  /// Interaction logic for MainWindow.xaml
  /// </summary>
  public partial class MainWindow : Window
  {
    #region CONST

    public const int NORM_IMG_SIZE = 48;

    #endregion

    public MainWindow()
    {
      InitializeComponent();

      initNet();
    }

    private ConvNet m_Network;

    #region Init

    private void initNet()
    {
      try
      {
        var assembly = Assembly.GetExecutingAssembly();
        using (var stream = assembly.GetManifestResourceStream("ML.MainColorDemo.net.mld"))
        {
          m_Network = ConvNet.Deserialize(stream);
          m_Network.IsTraining = false;
        }
      }
      catch (Exception error)
      {
        MessageBox.Show("Error while CNN deserialize: "+error.Message);
      }
    }

    #endregion

    #region Load Image

    private void onUploadButtonClick(object sender, RoutedEventArgs e)
    {
      var dialog = new Microsoft.Win32.OpenFileDialog();
      dialog.Filter = "Image Files (*.jpg; *.jpeg; *.gif; *.bmp; *.png)|*.jpg; *.jpeg; *.gif; *.bmp; *.png";
      if (!(bool)dialog.ShowDialog()) return;

      processImage(dialog.FileName, null);
    }

    private void onImageDrop(object sender, DragEventArgs e)
    {
      processImage(null, e.Data);
    }

    private void processImage(string path, IDataObject data)
    {
      try
      {
        //using (var bitmap = loadImage(path, data))
        //using (var normBitmap = doNormalization(bitmap, true))
        //  doRecognition(normBitmap);

        using (var bitmap = loadImage(path, data))
        using (var normBitmap = doDowngrade(bitmap))
          doDirect(normBitmap);
      }
      catch (Exception ex)
      {
        MessageBox.Show("Error: " + ex.Message);
      }
    }

    private Bitmap loadImage(string path, IDataObject data)
    {
      Bitmap result = null;
      if (!string.IsNullOrWhiteSpace(path))
      {
        result = new Bitmap(path);
      }
      else if (data.GetDataPresent(DataFormats.FileDrop))
      {
        var files = (string[])data.GetData(DataFormats.FileDrop);
        if (files.Any())
          result = new Bitmap(files[0]);
      }
      else
      {
        var html = (string)data.GetData(DataFormats.Html);
        var anchor = "src=\"";
        int idx1 = html.IndexOf(anchor) + anchor.Length;
        int idx2 = html.IndexOf("\"", idx1);
        var url = html.Substring(idx1, idx2 - idx1);

        if (url.StartsWith("http"))
        {
          using (var client = new WebClient())
          {
            var bytes = client.DownloadData(url);
            using (var stream = new MemoryStream(bytes))
            {
              result = new Bitmap(stream);
            }
          }
        }
        else if (url.StartsWith("data:image"))
        {
          anchor = "base64,";
          idx1 = url.IndexOf(anchor) + anchor.Length;
          var base64Data = url.Substring(idx1);
          var bytes = Convert.FromBase64String(base64Data);
          using (var stream = new MemoryStream(bytes))
          {
            result = new Bitmap(stream);
          }
        }
      }

      if (result == null)
      {
        throw new Exception("Cannot load image");
      }

      m_DropHereTxt.Visibility = Visibility.Collapsed;
      m_Border.Visibility  = Visibility.Visible;
      m_ImgInitial.Source = imageSourceFromBitmap(result);

      return result;
    }

    #endregion

    #region Normalization

    private Bitmap doNormalization(Bitmap bitmap, bool show)
    {
      // crop image to center square size
      // and normalize image to NORM_IMG_SIZE x NORM_IMG_SIZE

      var xm = bitmap.Width;
      var ym = bitmap.Height;
      var cm = Math.Min(xm, ym);
      var normBitmap = new Bitmap(NORM_IMG_SIZE, NORM_IMG_SIZE);

      using (var gr = Graphics.FromImage(normBitmap))
      {
        gr.DrawImage(bitmap,
                     new Rectangle(0, 0, NORM_IMG_SIZE, NORM_IMG_SIZE),
                     new Rectangle((xm - cm) / 2, (ym - cm) / 2, cm, cm),
                     GraphicsUnit.Pixel);
      }

      if (show)
        m_ImgNormalized.Source = imageSourceFromBitmap(normBitmap);

      return normBitmap;
    }

    #endregion

    #region Recognition

    private void doRecognition(Bitmap normBitmap)
    {
      if (normBitmap == null)
      {
        MessageBox.Show("Upload image first");
        return;
      }

      m_ResultsPanel.Visibility = Visibility.Visible;

      for (int i=0; i<4; i++)
      {
        var barName = string.Format("m_Color{0}", i+1);
        ((System.Windows.Shapes.Rectangle)this.FindName(barName)).Fill = System.Windows.Media.Brushes.Transparent;
      }

      var data = bitmapToInput(normBitmap);
      var result = m_Network.Calculate(data);

      for (int i=0; i<12; i+=3)
      {
        var r = (byte)(Math.Min(result[i][0,0]*255, 255));
        var g = (byte)(Math.Min(result[i+1][0,0]*255, 255));
        var b = (byte)(Math.Min(result[i+2][0,0]*255, 255));
        var color = System.Windows.Media.Color.FromRgb(r, g, b);
        var barName = string.Format("m_Color{0}", i/3+1);
        ((System.Windows.Shapes.Rectangle)this.FindName(barName)).Fill = new SolidColorBrush(color);
      }
    }

    #endregion

    #region Direct

    private void doDirect(Bitmap bitmap)
    {
      //const int factor = 63;
      var kk = 0.3F;

      var kernel = new Func<float, float>(r => { if (r<kk) return 1.0F; if (r>1) return 0.0F; return (1.0F-r)/(1-kk); });
      var hist = new Dictionary<System.Drawing.Color, float>();
      var h = bitmap.Height;
      var w = bitmap.Width;
      for (int x=0; x<w; x++)
      for (int y=0; y<h; y++)
      {
        var p = bitmap.GetPixel(x, y);
        //var qp = System.Drawing.Color.FromArgb(factor*(p.R/factor), factor*(p.G/factor), factor*(p.B/factor));

        var xi  = (2.0F*x-w)/w;
        var eta = (2.0F*y-h)/h;
        var r = xi*xi + eta*eta;
        var weight = kernel(r);
        if (weight<=0) continue;

        if (!hist.ContainsKey(p)) hist[p] = 0.0F;
        hist[p] += weight;
      }

      var ccount = 3;
      var topColors = hist.OrderByDescending(c => c.Value)
                          .Take(ccount)
                          .Select(c => c.Key)
                          .ToList();

      for (int i=0; i<3; i++)
      {
        var tcolor = topColors[i];
        var color = System.Windows.Media.Color.FromRgb(tcolor.R, tcolor.G, tcolor.B);
        var barName = string.Format("m_Color{0}", i+1);
        ((System.Windows.Shapes.Rectangle)this.FindName(barName)).Fill = new SolidColorBrush(color);
      }
    }

    private Bitmap doDowngrade(Bitmap bitmap)
    {
      const int factor = 25;

      var result = new Bitmap(bitmap.Width, bitmap.Height);
      for (int x=0; x<bitmap.Width; x++)
      for (int y=0; y<bitmap.Height; y++)
      {
        var p = bitmap.GetPixel(x, y);
        var d = System.Drawing.Color.FromArgb(factor*(p.R/factor), factor*(p.G/factor), factor*(p.B/factor));
        result.SetPixel(x, y, d);
      }

      m_ImgNormalized.Source = imageSourceFromBitmap(result);

      return result;
    }

    #endregion

    #region Utility

    private ImageSource imageSourceFromBitmap(Bitmap bitmap)
    {
      var ip = bitmap.GetHbitmap();
      BitmapSource bs = null;
      try
      {
        bs = System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(
               ip,
               IntPtr.Zero, Int32Rect.Empty,
               BitmapSizeOptions.FromEmptyOptions());
      }
      finally
      {
        DeleteObject(ip);
      }

      return bs;
    }

    [DllImport("gdi32")]
    static extern int DeleteObject(IntPtr o);

    private Bitmap imageSourceToBitmap(BitmapSource srs)
    {
      var width = srs.PixelWidth;
      var height = srs.PixelHeight;
      var stride = width * ((srs.Format.BitsPerPixel + 7) / 8);
      var ptr = IntPtr.Zero;

      try
      {
        ptr = Marshal.AllocHGlobal(height * stride);
        srs.CopyPixels(new Int32Rect(0, 0, width, height), ptr, height * stride, stride);
        using (var btm = new Bitmap(width, height, stride, System.Drawing.Imaging.PixelFormat.Format1bppIndexed, ptr))
        {
          // Clone the bitmap so that we can dispose it and
          // release the unmanaged memory at ptr
          return new Bitmap(btm);
        }
      }
      finally
      {
        if (ptr != IntPtr.Zero)
          Marshal.FreeHGlobal(ptr);
      }
    }

    private double[][,] bitmapToInput(Bitmap bitmap)
    {
      var data = new double[3][,]
                 {
                   new double[NORM_IMG_SIZE, NORM_IMG_SIZE],
                   new double[NORM_IMG_SIZE, NORM_IMG_SIZE],
                   new double[NORM_IMG_SIZE, NORM_IMG_SIZE]
                 };
      for (int y = 0; y < NORM_IMG_SIZE; y++)
      for (int x = 0; x < NORM_IMG_SIZE; x++)
      {
        var p = bitmap.GetPixel(x, y);
        data[0][y, x] = p.R / 255.0D;
        data[1][y, x] = p.G / 255.0D;
        data[2][y, x] = p.B / 255.0D;
      }

      return data;
    }

    #endregion
  }
}
