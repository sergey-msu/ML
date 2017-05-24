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
using System.Drawing.Drawing2D;
using ML.DeepMethods.Algorithms;

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
    private ConvNet m_NetworkF;
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
        using (var stream = assembly.GetManifestResourceStream("ML.CatDogDemo.cat-dog-filt-19.28.mld"))
        {
          m_NetworkF = ConvNet.Deserialize(stream);
          m_NetworkF.IsTraining = false;
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
          doRecognition(bitmap, normBitmap);
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

    private Class doRecognition(Bitmap bitmap, Bitmap normBitmap)
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

      var data1 = bitmapToInput(normBitmap);
      var data2 = getNetFData(bitmap);
      var start = DateTime.Now;

      var alp1 = 0.7D;
      var alp2 = 0.3D;
      var result1 = m_Network.Calculate(data1);
      var result2 = m_NetworkF.Calculate(data2);
      var result  = new double[result1.Length];
      result[0] = alp1*result1[0][0,0] + (1-alp1)*result2[0][0,0];
      result[1] = alp2*result1[1][0,0] + (1-alp2)*result2[1][0,0];


      var end = DateTime.Now;
      m_PredictionTime.Text = string.Format("{0} ms", (int)(end-start).TotalMilliseconds);

      var max = double.MinValue;
      var idx = -1;
      var total = 0.0D;
      for (int i = 0; i < m_Classes.Count; i++) { total += result[i]; }
      if (total <= 0)
      {
        m_TxtResult.Text = "?";
        return Class.Unknown;
      }

      for (int i = 0; i < m_Classes.Count; i++)
      {
        var prob = Math.Round(result[i] / total, 2);
        var barName = string.Format("m_Bar{0}", i);
        var probName = string.Format("m_Prob{0}", i);
        ((System.Windows.Shapes.Rectangle)this.FindName(barName)).Width = prob * 30;
        ((TextBlock)this.FindName(probName)).Text = prob.ToString();

        if (result[i] > max)
        {
          max = result[i];
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
      var errors1 = 0;
      var errors1C = 0;
      var errors1D = 0;
      var errors2 = 0;
      var errors2C = 0;
      var errors2D = 0;
      var errorsC1 = 0;
      var errorsC2 = 0;
      var errorsR = 0;
      var pct1 = 0;
      var pct1C = 0;
      var pct1D = 0;
      var pct2 = 0;
      var pct2C = 0;
      var pct2D = 0;
      var pctC = 0;
      var pctC1 = 0;
      var pctC2 = 0;
      var pctR = 0;
      var alp1 = 0.95D;
      var alp2 = 0.05D;
      var dir = new DirectoryInfo(path);
      var total = dir.GetFiles().Length;

      var sample = new MultiRegressionSample<double[][,]>();
      var cat = new double[] { 1.0D, 0.0D };
      var dog = new double[] { 0.0D, 1.0D };

      int cnt = 0;
      foreach (var file in dir.EnumerateFiles().Shuffle(0).Skip(10000).Take(500))
      {
        var fname = Path.GetFileNameWithoutExtension(file.Name);
        var expected = fname.StartsWith("cat.") ? 0 : 1;
        var data1 = getNetData(file.FullName);
        double[][,] data2;
        using (var image = (Bitmap)System.Drawing.Image.FromFile(file.FullName))
          data2 = getNetFData(image);

        sample.Add(data2, expected==0 ? cat : dog);

        var result1 = m_Network.Calculate(data1).Select(d => d[0,0]).ToArray();
        var actual1 = ML.Core.Mathematics.MathUtils.ArgMax(result1);
        if (expected != actual1)
        {
          if (expected==0) errors1C++;
          else errors1D++;
          errors1++;
        }

        var result2 = m_NetworkF.Calculate(data2).Select(d => d[0,0]).ToArray();
        var actual2 = ML.Core.Mathematics.MathUtils.ArgMax(result2);
        if (expected != actual2)
        {
          if (expected==0) errors2C++;
          else errors2D++;
          errors2++;
        }

        var resultR = new double[result1.Length];
        resultR[0] = alp1*result1[0] + (1-alp1)*result2[0];
        resultR[1] = alp2*result1[1] + (1-alp2)*result2[1];
        var actualR = ML.Core.Mathematics.MathUtils.ArgMax(resultR);
        if (expected != actualR) errorsR++;

        if ((expected != actual1) && (expected != actual2))
        {
          if (expected==0) errorsC1++;
          else errorsC2++;
        }

        cnt++;
        pct1  = errors1*100/cnt;
        pct2  = errors2*100/cnt;
        pctC1 = errorsC1*100/cnt;
        pctC2 = errorsC2*100/cnt;
        pctC  = (errorsC1+errorsC2)*100/cnt;
        pctR  = errorsR*100/cnt;
      }

      var alg = new BackpropAlgorithm(m_NetworkF);
      var err = alg.GetErrors(sample, 0, true);

      var message = "Errors1: {0}%, Errors2: {1}%, ErrorsC: {2}%, ErrorR: {3}%";
      MessageBox.Show(string.Format(message, pct1, pct2, pctC, pctR));
    }

    private double[][,] getNetData(string fpath)
    {
      using (var bitmap = new Bitmap(fpath))
      using (var normBitmap = doNormalization(bitmap, false))
      using (var grayBitmap = makeGrayscale3v0(normBitmap))
      {
        return bitmapToInput(grayBitmap);
      }
    }

    private double[][,] getNetFData(Bitmap image)
    {
      using (var filtImage = Utils.Filters.Sobel3x3Filter(image))
      using (var normImage = new Bitmap(48, 48))
      using (var normFiltImage = new Bitmap(48, 48))
      {
        var w = image.Width;
        var h = image.Height;
        var s = Math.Min(w, h);

        // crop image to center square size
        // and normalize image to NORM_IMG_SIZE x NORM_IMG_SIZE

        using (var gr = Graphics.FromImage(normImage))
        {
          gr.InterpolationMode  = InterpolationMode.HighQualityBicubic;
          gr.CompositingQuality = CompositingQuality.HighQuality;
          gr.SmoothingMode      = SmoothingMode.AntiAlias;

          gr.DrawImage(image,
                       new Rectangle(0, 0, 48, 48),
                       new Rectangle((w - s) / 2, (h - s) / 2, s, s),
                       GraphicsUnit.Pixel);
        }
        using (var gr = Graphics.FromImage(normFiltImage))
        {
          gr.InterpolationMode  = InterpolationMode.HighQualityBicubic;
          gr.CompositingQuality = CompositingQuality.HighQuality;
          gr.SmoothingMode      = SmoothingMode.AntiAlias;

          gr.DrawImage(filtImage,
                       new Rectangle(0, 0, 48, 48),
                       new Rectangle((w - s) / 2, (h - s) / 2, s, s),
                       GraphicsUnit.Pixel);
        }

        // digitize images

        var result = new double[2][,]
        {
          new double[48, 48],
          new double[48, 48]
        };

        // grayscale
        for (var y=0; y<48; y++)
        for (var x=0; x<48; x++)
        {
          var pixel = normImage.GetPixel(x, y);
          var level = (pixel.R + pixel.G + pixel.B) / (3*255.0D);
          result[0][y, x] = level;
        }

        // filter
        for (var y=0; y<48; y++)
        for (var x=0; x<48; x++)
        {
          var pixel = normFiltImage.GetPixel(x, y);
          var level = (pixel.R + pixel.G + pixel.B) / (3*255.0D);
          result[1][y, x] = level;
        }

        return result;
      }
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
