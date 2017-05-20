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

namespace ML.CatDogDemo
{
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

      initNet();
    }

    private ConvNet m_Network;
    private Dictionary<int, Class> m_Classes = new Dictionary<int, Class>()
    {
      { 3, new Class("Cat",        0) },
      { 5, new Class("Dog",        1) },
    };

    #region Init

    private void initNet()
    {
      try
      {
        var assembly = Assembly.GetExecutingAssembly();
        using (var stream = assembly.GetManifestResourceStream("ML.CatDogDemo.cat-dog-19.1.mld"))
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
        using (var normBitmap = doNormalization(bitmap))
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

    private Bitmap doNormalization(Bitmap bitmap)
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
        return Class.None;
      }

      m_ResultsPanel.Visibility = Visibility.Visible;

      for (int i = 0; i < m_Classes.Count; i++)
      {
        var barName = string.Format("m_Bar{0}", i);
        var probName = string.Format("m_Prob{0}", i);
        ((System.Windows.Shapes.Rectangle)this.FindName(barName)).Width = 0;
        ((TextBlock)this.FindName(probName)).Text = "";
      }

      var data = new double[3][,]
                 {
                   new double[NORM_IMG_SIZE, NORM_IMG_SIZE],
                   new double[NORM_IMG_SIZE, NORM_IMG_SIZE],
                   new double[NORM_IMG_SIZE, NORM_IMG_SIZE]
                 };
      for (int y = 0; y < NORM_IMG_SIZE; y++)
        for (int x = 0; x < NORM_IMG_SIZE; x++)
        {
          var p = normBitmap.GetPixel(x, y);
          data[0][y, x] = p.R / 255.0D;
          data[1][y, x] = p.R / 255.0D;
          data[2][y, x] = p.R / 255.0D;
        }

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
        return Class.None;
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

    #endregion

    private void ArchitectureButton_Click(object sender, RoutedEventArgs e)
    {
      var details = new ArchitectureWindow();
      details.ShowDialog();
    }
  }
}
