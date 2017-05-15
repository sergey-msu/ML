using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;

using ML.Core;
using ML.DeepMethods.Models;
using System.IO;
using System.Net;

namespace PicturePrimitive
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

    private Bitmap m_Bitmap;
    private Bitmap m_NormalizedBitmap;
    private ConvNet m_Network;
    private Dictionary<int, Class> m_Classes = new Dictionary<int, Class>()
    {
      //{ 0, new Class("Airplane",   ) },
      //{ 1, new Class("Automobile", ) },
      //{ 2, new Class("Bird",       ) },
        { 3, new Class("Cat",        0) },
      //{ 4, new Class("Deer",       ) },
        { 5, new Class("Dog",        1) },
      //{ 6, new Class("Frog",       ) },
      //{ 7, new Class("Horse",      2) },
      //{ 8, new Class("Ship",       ) },
      //{ 9, new Class("Truck",      ) },
    };

    private void initNet()
    {
      var assembly = Assembly.GetExecutingAssembly();
      using (var stream = assembly.GetManifestResourceStream("ML.CifarTruncDemo.net.mld"))
      {
        m_Network = ConvNet.Deserialize(stream);
        m_Network.IsTraining = false;
      }
    }

    #region Load Image

    private void onUploadButtonClick(object sender, RoutedEventArgs e)
    {
      try
      {
        var dialog = new Microsoft.Win32.OpenFileDialog();
        dialog.Filter = "Image Files (*.jpg; *.jpeg; *.gif; *.bmp; *.png)|*.jpg; *.jpeg; *.gif; *.bmp; *.png";

        if ((bool)dialog.ShowDialog())
        {
          loadImage(dialog.FileName, null);
          doNormalization();
          doRecognition();
        }
      }
      catch (Exception ex)
      {
        MessageBox.Show("Error: " + ex.Message);
      }
    }

    private void onImageDrop(object sender, DragEventArgs e)
    {
      try
      {
        loadImage(null, e.Data);
        doNormalization();
        doRecognition();
      }
      catch (Exception ex)
      {
        MessageBox.Show("Error: " + ex.Message);
      }
    }

    private void loadImage(string path, IDataObject data)
    {
      if (!string.IsNullOrWhiteSpace(path))
      {
        m_Bitmap = new Bitmap(path);
      }
      else if (data.GetDataPresent(DataFormats.FileDrop))
      {
        var files = (string[])data.GetData(DataFormats.FileDrop);
        if (files.Any())
          m_Bitmap = new Bitmap(files[0]);
      }
      else
      {
        var html = (string)data.GetData(DataFormats.Html);
        var anchor = "src=\"";
        int idx1 = html.IndexOf(anchor) + anchor.Length;
        int idx2 = html.IndexOf("\"", idx1);
        var url = html.Substring(idx1, idx2 - idx1);

        var src = new BitmapImage();
        using (var client = new WebClient())
        {
          var bytes = client.DownloadData(url);
          using (var stream = new MemoryStream(bytes))
          {
            src.BeginInit();
            src.CacheOption = BitmapCacheOption.OnLoad;
            src.StreamSource = stream;
            src.EndInit();
            src.Freeze();
          }
        }

        m_ImgInitial.Source = src;
        return;
      }

      if (m_Bitmap == null)
        MessageBox.Show("Cannot load image");

      m_DropHereTxt.Visibility = Visibility.Collapsed;
      m_ImgInitial.Visibility = Visibility.Visible;
      m_ImgInitial.Source = imageSourceFromBitmap(m_Bitmap);
    }

    #endregion

    #region Normalization

    private void doNormalization()
    {
      // crop image to center square size
      // and normalize image to NORM_IMG_SIZE x NORM_IMG_SIZE

      var xm = m_Bitmap.Width;
      var ym = m_Bitmap.Height;
      var cm = Math.Min(xm, ym);
      m_NormalizedBitmap = new Bitmap(NORM_IMG_SIZE, NORM_IMG_SIZE);

      using (var gr = Graphics.FromImage(m_NormalizedBitmap))
      {
        gr.DrawImage(m_Bitmap,
                     new Rectangle(0, 0, NORM_IMG_SIZE, NORM_IMG_SIZE),
                     new Rectangle((xm - cm) / 2, (ym - cm) / 2, cm, cm),
                     GraphicsUnit.Pixel);
      }

      m_ImgNormalized.Source = imageSourceFromBitmap(m_NormalizedBitmap);
    }

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

    #region Recognition

    private void doRecognition()
    {
      if (m_NormalizedBitmap == null)
      {
        MessageBox.Show("Upload image first");
        return;
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
          var p = m_NormalizedBitmap.GetPixel(x, y);
          data[0][y, x] = p.R / 255.0D;
          data[1][y, x] = p.R / 255.0D;
          data[2][y, x] = p.R / 255.0D;
        }

      var result = m_Network.Calculate(data);

      var max = double.MinValue;
      var idx = -1;
      var total = 0.0D;
      for (int i = 0; i < m_Classes.Count; i++) { total += result[i][0, 0]; }
      if (total <= 0)
      {
        m_TxtResult.Text = "?";
        return;
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
    }

    #endregion
  }
}
