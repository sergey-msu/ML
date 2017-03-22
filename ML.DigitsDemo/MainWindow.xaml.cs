using ML.Core.Mathematics;
using ML.DeepMethods.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace ML.DigitsDemo
{
  /// <summary>
  /// Interaction logic for MainWindow.xaml
  /// </summary>
  public partial class MainWindow : Window
  {
    private Point m_CurrentPoint = new Point();
    private ConvolutionalNetwork m_Network;


    public MainWindow()
    {
      InitializeComponent();

      initNet();

      m_Canvas.MouseDown   += canvas_MouseDown;
      m_Canvas.MouseMove   += canvas_MouseMove;
      m_BtnClear.Click     += btnClear_Click;
      m_BtnRecognize.Click += btnRecognize_Click;
    }

    private void initNet()
    {
      var assembly = Assembly.GetExecutingAssembly();
      using (var stream = assembly.GetManifestResourceStream("ML.DigitsDemo.lenet1.mld"))
      {
        m_Network = ConvolutionalNetwork.Deserialize(stream);
      }
    }

    #region Canvas

    private void btnClear_Click(object sender, RoutedEventArgs e)
    {
      m_Canvas.Children.Clear();
    }

    private void canvas_MouseDown(object sender, MouseButtonEventArgs e)
    {
      if (e.ButtonState == MouseButtonState.Pressed)
        m_CurrentPoint = e.GetPosition(m_Canvas);
    }

    private void canvas_MouseMove(object sender, MouseEventArgs e)
    {
      var half = 3.5D;
      var thickness = 2*half;

      if (e.LeftButton == MouseButtonState.Pressed)
      {
        Line line = new Line
        {
          Stroke = Brushes.Black,
          StrokeThickness = thickness,
          StrokeDashCap = PenLineCap.Round,
          StrokeStartLineCap = PenLineCap.Round,
          StrokeEndLineCap = PenLineCap.Round
        };

        line.X1 = Math.Min(Math.Max(half, m_CurrentPoint.X), m_Canvas.Width-half);
        line.Y1 = Math.Min(Math.Max(half, m_CurrentPoint.Y), m_Canvas.Height-half);
        line.X2 = Math.Min(Math.Max(half, e.GetPosition(m_Canvas).X), m_Canvas.Width-half);
        line.Y2 = Math.Min(Math.Max(half, e.GetPosition(m_Canvas).Y), m_Canvas.Height-half);

        m_CurrentPoint = e.GetPosition(m_Canvas);

        m_Canvas.Children.Add(line);
      }
    }

    #endregion

    #region Recognize

    private void btnRecognize_Click(object sender, RoutedEventArgs e)
    {
      var snapshot = takeCanvasSnapshot();
      var data = normalizeData(snapshot);
      if (data==null)
      {
        m_TbResult.Text = "?";
        return;
      }

      //data = new double[1,28,28]
      //{
      //  {
      //    { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
      //    { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
      //    { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
      //    { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
      //    { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
      //    { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
      //    { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
      //    { 0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0 },
      //    { 0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0 },
      //    { 0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0 },
      //    { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0 },
      //    { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0 },
      //    { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0 },
      //    { 0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0 },
      //    { 0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0 },
      //    { 0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0 },
      //    { 0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0 },
      //    { 0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,1,1,0,0,0,0 },
      //    { 0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0 },
      //    { 0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0 },
      //    { 0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0 },
      //    { 0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
      //    { 0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
      //    { 0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
      //    { 0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
      //    { 0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
      //    { 0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
      //    { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 }
      //  }
      //};



      var result = m_Network.Calculate(data);
      var prob = 0.0D;
      var digit = -1;
      for (int i=0; i<9; i++)
      {
        if (result[i,0,0]>prob)
        {
          prob=result[i,0,0];
          digit=i;
        }
      }

      m_TbResult.Text = (digit < 0) ? "?" : digit.ToString();
    }

    private double[,] takeCanvasSnapshot()
    {
      var bounds = VisualTreeHelper.GetDescendantBounds(m_Canvas);
      var rtb = new RenderTargetBitmap((int)bounds.Width, (int)bounds.Height, 96, 96, PixelFormats.Pbgra32);
      var dv = new DrawingVisual();
      using (var dc = dv.RenderOpen())
      {
          var vb = new VisualBrush(m_Canvas);
          dc.DrawRectangle(vb, null, new Rect(new Point(), bounds.Size));
      }
      rtb.Render(dv);

      var xlen = (int)m_Canvas.ActualWidth;
      var ylen = (int)m_Canvas.ActualHeight;
      var result = new double[ylen, xlen];

      int stride = rtb.PixelWidth * 4;
      int size = rtb.PixelHeight * stride;
      byte[] pixels = new byte[size];
      rtb.CopyPixels(pixels, stride, 0);

      for(int y=0; y<m_Canvas.ActualHeight; y++)
      for(int x=0; x<m_Canvas.ActualWidth; x++)
      {
        int index = y*stride + 4*x;
        byte red   = pixels[index];
        byte green = pixels[index+1];
        byte blue  = pixels[index+2];
        byte alpha = pixels[index+3];

        var shade = 255-red;
        result[y, x] = shade/255.0D;
      }

      return result;
    }

    private double[,,] normalizeData(double[,] data)
    {
      // extract frame

      var xlen = data.GetLength(1);
      var ylen = data.GetLength(0);

      var xmin = 0;
      for (int x=0; x<xlen; x++)
      {
        bool empty = true;
        for (int y=0; y<ylen; y++)
          if (data[y,x] > 0) { empty=false; break; }
        if (!empty) break;
        xmin++;
      }

      var xmax = xlen-1;
      for (int x=xlen-1; x>=0; x--)
      {
        bool empty = true;
        for (int y=0; y<ylen; y++)
          if (data[y,x] > 0) { empty=false; break; }
        if (!empty) break;
        xmax--;
      }

      if (xmin >= xmax) return null;

      var ymin = 0;
      for (int y=0; y<ylen; y++)
      {
        bool empty = true;
        for (int x=0; x<ylen; x++)
          if (data[y,x] > 0) { empty=false; break; }
        if (!empty) break;
        ymin++;
      }

      var ymax = ylen-1;
      for (int y=ylen-1; y>=0; y--)
      {
        bool empty = true;
        for (int x=0; x<xlen; x++)
          if (data[y,x] > 0) { empty=false; break; }
        if (!empty) break;
        ymax--;
      }

      if (ymin >= ymax) return null;

      // find mean point

      var xsum = 0.0D;
      var ysum = 0.0D;
      var mass = 0.0D;
      for (int y=ymin; y<=ymax; y++)
      for (int x=xmin; x<=xmax; x++)
      {
        xsum += x*data[y, x];
        ysum += y*data[y, x];
        mass += data[y, x];
      }

      var xmean = xsum / mass;
      var ymean = ysum / mass;

      // scale result

      var lambda = (ymax-ymin)/20.0D;
      var result = new double[1, 28, 28];

      for (var v=0; v<28; v++)
      for (var u=0; u<28; u++)
      {
        var xidx = (u-13.5D)*lambda + xmean;
        if (xidx<0 || xidx>=xlen) { result[0, v, u] = 0; continue; }

        var yidx = (v-13.5D)*lambda + ymean;
        if (yidx<0 || yidx>=ylen) { result[0, v, u] = 0; continue; }

        var xm = (int)xidx;
        var xM = (xm < xlen-1) ? xm+1 : xm;
        var ym = (int)yidx;
        var yM = (ym < ylen-1) ? ym+1 : ym;
        var d00 = data[ym, xm];
        var d01 = data[ym, xM];
        var d10 = data[yM, xm];
        var d11 = data[yM, xM];

        result[0, v, u] = (d11+d00-d01-d10)*(xidx-xm)*(yidx-ym) +
                          (d01-d00)*(xidx-xm) +
                          (d10-d00)*(yidx-ym) +
                          d00;
      }

      return result;
    }

    #endregion
  }
}
