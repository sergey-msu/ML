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
    private Dictionary<int, Class> m_Classes = new Dictionary<int, Class>()
    {
      { 3, new Class("Cat", 0) },
      { 5, new Class("Dog", 1) },
    };

    #region Init

    private void initNet()
    {
      try
      {
        var assembly = Assembly.GetExecutingAssembly();
        using (var stream = assembly.GetManifestResourceStream("ML.CatDogDemo.cat-dog-20.2.mld"))
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
        using (var bitmap = loadImage(path, data))
        using (var normBitmap = doNormalization(bitmap, true))
          doRecognition(normBitmap);
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

    private Class doRecognition(Bitmap normBitmap)
    {
      if (normBitmap == null)
      {
        MessageBox.Show("Upload image first");
        return Class.Unknown;
      }

      m_ResultsPanel.Visibility = Visibility.Visible;

      for (int i = 0; i < m_Classes.Count; i++)
      {
        var barName = string.Format("m_Bar{0}", i);
        var probName = string.Format("m_Prob{0}", i);
        ((System.Windows.Shapes.Rectangle)this.FindName(barName)).Width = 0;
        ((TextBlock)this.FindName(probName)).Text = "";
      }

      var data = bitmapToInput(normBitmap);
      var start = DateTime.Now;
      var result = m_Network.Calculate(data);
      var end = DateTime.Now;
      m_PredictionTime.Text = string.Format("{0} ms", (int)(end-start).TotalMilliseconds);

      var max = double.MinValue;
      var idx = -1;
      var total = 0.0D;
      for (int i = 0; i < m_Classes.Count; i++) { total += result[i][0, 0]; }
      if (total <= 0)
      {
        m_TxtResult.Text = "?";
        return Class.Unknown;
      }

      for (int i = 0; i < m_Classes.Count; i++)
      {
        var prob = Math.Round(result[i][0, 0] / total, 2);
        var barName = string.Format("m_Bar{0}", i);
        var probName = string.Format("m_Prob{0}", i);
        ((System.Windows.Shapes.Rectangle)this.FindName(barName)).Width = prob * 30;
        ((TextBlock)this.FindName(probName)).Text = prob.ToString();

        if (result[i][0, 0] > max)
        {
          max = result[i][0, 0];
          idx = i;
        }
      }

      var cls = m_Classes.First(c => c.Value.Value == idx);

      m_TxtResult.Text = cls.Value.Name;

      return cls.Value;
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
      var data = new double[1][,]
                 {
                   new double[NORM_IMG_SIZE, NORM_IMG_SIZE]
                 };
      for (int y = 0; y < NORM_IMG_SIZE; y++)
      for (int x = 0; x < NORM_IMG_SIZE; x++)
      {
        var p = bitmap.GetPixel(x, y);
        var level = (p.R + p.G + p.B) / 3.0D;
        data[0][y, x] = level / 255.0D;
      }

      return data;
    }

    #endregion

    private void ArchitectureButton_Click(object sender, RoutedEventArgs e)
    {
      var details = new ArchitectureWindow();
      details.ShowDialog();
    }

    #region checks

    private void onTestButtonClick(object sender, RoutedEventArgs e)
    {
      var path = @"F:\Work\science\Machine learning\data\cat-dog\train\kaggle";
      var errors = 0;
      var dir = new DirectoryInfo(path);
      var total = dir.GetFiles().Length;

      foreach (var file in dir.EnumerateFiles().Skip(10000).Take(4000))
      {
        var fname = Path.GetFileNameWithoutExtension(file.Name);
        var expected = fname.StartsWith("cat.") ? 0 : 1;
        using (var bitmap = new Bitmap(file.FullName))
        using (var normBitmap = doNormalization(bitmap, false))
        using (var grayBitmap = makeGrayscale3v0(normBitmap))
        {
          var data = bitmapToInput(grayBitmap);
          var result = m_Network.Calculate(data).Select(d => d[0,0]).ToArray();
          var actual = ML.Core.Mathematics.MathUtils.ArgMax<double>(result);
          if (expected != actual) errors++;
        }
      }

      MessageBox.Show("Errors: "+errors+" from 4000 ("+(errors*100/4000)+"%)");
    }

    private Bitmap makeGrayscale3v0(Bitmap bitmap)
    {
      var result = new Bitmap(bitmap.Width, bitmap.Height);
      for (int x = 0; x < bitmap.Width; x++)
      for (int y = 0; y < bitmap.Height; y++)
      {
        var p = bitmap.GetPixel(x, y);
        var level = (byte)((p.R + p.G + p.B) / 3);
        var grayColor = System.Drawing.Color.FromArgb(level, level, level);
        result.SetPixel(x, y, grayColor);
      }
      return result;
    }

    private Bitmap makeGrayscale3(Bitmap original)
    {
       //create a blank bitmap the same size as original
       Bitmap newBitmap = new Bitmap(original.Width, original.Height);

       //get a graphics object from the new image
       Graphics g = Graphics.FromImage(newBitmap);

       //create the grayscale ColorMatrix
       var colorMatrix = new ColorMatrix(
          new float[][]
          {
             new float[] {.3f, .3f, .3f, 0, 0},
             new float[] {.59f, .59f, .59f, 0, 0},
             new float[] {.11f, .11f, .11f, 0, 0},
             new float[] {0, 0, 0, 1, 0},
             new float[] {0, 0, 0, 0, 1}
          });

       //create some image attributes
       var attributes = new ImageAttributes();

       //set the color matrix attribute
       attributes.SetColorMatrix(colorMatrix);

       //draw the original image on the new image
       //using the grayscale color matrix
       g.DrawImage(original, new Rectangle(0, 0, original.Width, original.Height),
          0, 0, original.Width, original.Height, GraphicsUnit.Pixel, attributes);

       //dispose the Graphics object
       g.Dispose();
       return newBitmap;
    }

    #endregion
  }
}
