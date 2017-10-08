using System;
using System.Linq;
using System.Xml.Linq;
using System.Collections.Generic;
using System.Globalization;

namespace Math.ConsoleTest
{
  class Program
  {
    static void Main(string[] args)
    {
      doLoadRace();

      Console.WriteLine("DONE");
      //Console.ReadLine();
    }

    #region .pvt

    private static void doLoadRace()
    {
      var path = @"F:\Work\git\ML\solution\Demos\Math.ConsoleTest\path\caucasus.gpx";
      var doc = XDocument.Load(path);

      var lats = new List<double>();
      var lons = new List<double>();

      // load
      foreach (var pt in doc.Descendants().Where(d => d.Name.LocalName=="trkpt"))
      {
        var lat = pt.Attribute("lat").Value;
        lats.Add(double.Parse(lat, CultureInfo.InvariantCulture));

        var lon = pt.Attribute("lon").Value;
        lons.Add(double.Parse(lon, CultureInfo.InvariantCulture));
      }

      var xs = lats.ToArray();
      var ys = lons.ToArray();

      // kalman
      var n = xs.Length;
      var fxOut = new double[n];
      var bxOut = new double[n];
      var fyOut = new double[n];
      var byOut = new double[n];
      Utils.Filtering.Kalman(xs, fxOut, 0.001, 0.001);
      Utils.Filtering.Kalman(ys, fyOut, 0.001, 0.001);
      Utils.Filtering.Kalman(xs, bxOut, 0.001, 0.001, false);
      Utils.Filtering.Kalman(ys, byOut, 0.001, 0.001, false);
      for (int i=0; i<n; i++)
      {
        fxOut[i] = (fxOut[i] + bxOut[i])/2;
        fyOut[i] = (fyOut[i] + byOut[i])/2;
      }

      // smoothing
      var sm = 5;
      var fSKernel = new double[sm];
      for (int i=0; i<sm; i++) { fSKernel[i] = 1/(double)sm; }
      var fxSmooth = new double[n];
      var fySmooth = new double[n];
      Utils.Filtering.Smooth(fxOut, fSKernel, fxSmooth);
      Utils.Filtering.Smooth(fyOut, fSKernel, fySmooth);

      // curvature
      var curv = new double[n];
      Utils.Differential.Curvature(fxSmooth, fySmooth, curv);
      var kcurv = new double[n];
      Utils.Filtering.Kalman(curv, kcurv, 0.01, 0.001);

      // angles
      var angles = new double[n];
      Utils.Differential.Angles(fxSmooth, fySmooth, angles);
      var sum = 0.0D;
      for (int i=0; i<n; i++)
        sum += angles[i];

      using (var file = System.IO.File.Create(@"race.data.csv"))
      using (var writer = new System.IO.StreamWriter(file))
      {
        for (int i=0; i<n; i++)
        {
          var s = i/(double)n;
          writer.WriteLine("{0},{1},{2},{3},{4},{5},{4},{6},{4},{7}",
                            xs[i].ToString(CultureInfo.InvariantCulture),
                            ys[i].ToString(CultureInfo.InvariantCulture),
                            fxOut[i].ToString(CultureInfo.InvariantCulture),
                            fyOut[i].ToString(CultureInfo.InvariantCulture),
                            s.ToString(CultureInfo.InvariantCulture),
                            curv[i].ToString(CultureInfo.InvariantCulture),
                            kcurv[i].ToString(CultureInfo.InvariantCulture),
                            angles[i].ToString(CultureInfo.InvariantCulture));
        }
      }
    }

    private static void doAngles()
    {
      var n = 100000;
      var inputX = new double[n];
      var inputY = new double[n];

      for (int i=0; i<n; i++)
      {
        var x = i/(double)n;

        inputX[i] = 2*System.Math.Sin(2*System.Math.PI*x);
        inputY[i] = 2*System.Math.Cos(2*System.Math.PI*x);
      }

      var angles = new double[n];
      Utils.Differential.Angles(inputX, inputY, angles);
      var sum = 0.0D;
      for (int i=0; i<n; i++)
        sum += angles[i];

      using (var file = System.IO.File.Create(@"angles.data.csv"))
      using (var writer = new System.IO.StreamWriter(file))
      {
        for (int i=0; i<n; i++)
        {
          var x = inputX[i];
          var y = inputY[i];
          var a = angles[i];
          writer.WriteLine("{0},{1},{0},{2}", x.ToString(CultureInfo.InvariantCulture),
                                              y.ToString(CultureInfo.InvariantCulture),
                                              a.ToString(CultureInfo.InvariantCulture));
        }
      }
    }

    private static void doFirstDerivative()
    {
      var n = 1000;
      var inputX = new double[n];
      var inputY = new double[n];

      for (int i=0; i<n; i++)
      {
        var x = i/(double)n;
        x += x*x;
        inputX[i] = x;
        inputY[i] = 1 - 5*x + 20*x*x - 15*x*x*x;
        //inputY[i] = x*x;
      }

      var deriv = new double[n];
      Utils.Differential.Derivative(inputX, inputY, 1, deriv);

      using (var file = System.IO.File.Create(@"derivative.data.csv"))
      using (var writer = new System.IO.StreamWriter(file))
      {
        for (int i=0; i<n; i++)
        {
          var x = inputX[i];
          var y = inputY[i];
          var k = deriv[i];
          writer.WriteLine("{0},{1},{0},{2}", x.ToString(CultureInfo.InvariantCulture),
                                              y.ToString(CultureInfo.InvariantCulture),
                                              k.ToString(CultureInfo.InvariantCulture));
        }
      }
    }

    private static void doCurvature()
    {
      var n = 1000;
      var inputX = new double[n];
      var inputY = new double[n];

      for (int i=0; i<n; i++)
      {
        var x = i/(double)n;
        //inputX[i] = x;
        //inputY[i] = 1 - 5*x + 20*x*x - 15*x*x*x;
        //inputY[i] = x*x;

        inputX[i] = 2*System.Math.Sin(2*System.Math.PI*x);
        inputY[i] = 2*System.Math.Cos(2*System.Math.PI*x);
      }

      var curv = new double[n];
      Utils.Differential.Curvature(inputX, inputY, curv);

      using (var file = System.IO.File.Create(@"curvature.data.csv"))
      using (var writer = new System.IO.StreamWriter(file))
      {
        for (int i=0; i<n; i++)
        {
          var x = inputX[i];
          var y = inputY[i];
          var k = curv[i];
          writer.WriteLine("{0},{1},{0},{2}", x.ToString(CultureInfo.InvariantCulture),
                                              y.ToString(CultureInfo.InvariantCulture),
                                              k.ToString(CultureInfo.InvariantCulture));
        }
      }
    }

    private static void doKalmanFilter()
    {
      var n = 1000;
      var random = new Random();
      var a = 0.01D;

      var inputX = new double[n];
      var inputY = new double[n];
      var initY  = new double[n];
      for (int i=0; i<n; i++)
      {
        var x = i/(double)n;
        var y = (1 - x)*x;
        inputX[i] = x;
        var r = a*(2*random.NextDouble() - 1);
        inputY[i] = y + r;
        initY[i] = y;
      }

      // kalman
      var fKalman = new double[n];
      var bKalman = new double[n];
      Utils.Filtering.Kalman(inputY, fKalman, 0.01, 0.0001);
      Utils.Filtering.Kalman(inputY, bKalman, 0.01, 0.0001, false);
      for (int i=0; i<n; i++)
        fKalman[i] = (fKalman[i] + bKalman[i])/2;

      // smoothing
      var sm = 30;
      var fSKernel = new double[sm];
      for (int i=0; i<sm; i++) { fSKernel[i] = 1/(double)sm; }
      var fSmooth = new double[n];
      Utils.Filtering.Smooth(fKalman, fSKernel, fSmooth);

      // curvature
      var icurv = new double[n];
      var curv = new double[n];
      Utils.Differential.Curvature(inputX, fSmooth, curv);
      Utils.Differential.Curvature(inputX, initY, icurv);

      // angles

      var angles = new double[n];
      Utils.Differential.Angles(inputX, fSmooth, angles);
      var sum = 0.0D;
      for (int i=0; i<n; i++)
        sum += angles[i];

      var iangles = new double[n];
      Utils.Differential.Angles(inputX, initY, iangles);
      var isum = 0.0D;
      for (int i=0; i<n; i++)
        isum += iangles[i];

      using (var file = System.IO.File.Create(@"kalman.data.csv"))
      using (var writer = new System.IO.StreamWriter(file))
      {
        for (int i=0; i<n; i++)
        {
          var x = i/(double)n;
          writer.WriteLine("{0},{1},{0},{2},{0},{3},{0},{4},{0},{5},{0},{6},{0},{7}",
                             x.ToString(CultureInfo.InvariantCulture),
                             initY[i].ToString(CultureInfo.InvariantCulture),
                             inputY[i].ToString(CultureInfo.InvariantCulture),
                             fSmooth[i].ToString(CultureInfo.InvariantCulture),
                             curv[i].ToString(CultureInfo.InvariantCulture),
                             icurv[i].ToString(CultureInfo.InvariantCulture),
                             angles[i].ToString(CultureInfo.InvariantCulture),
                             iangles[i].ToString(CultureInfo.InvariantCulture));
        }
      }
    }

    private static void doLorenzSystem()
    {
      var sigma = 10.0D;
      var beta  = 8/3.0D;
      var rho   = 28.0D;

      var x0 = 0.0D;
      var x1 = 50.0D;
      var N  = 1000000;
      var h  = (x1 - x0)/N;
      var y0 = new double[] { 1, 1, 1.00001 };
      var buf = new double[3];
      var f  = new Func<double, double[], double[]>(
                 (x, y) =>
                 {
                   buf[0] = sigma*(y[1] - y[0]);
                   buf[1] = y[0]*(rho - y[2]) - y[1];
                   buf[2] = y[0]*y[1] - beta*y[2];
                   return buf;
                 });

      var r = new double[3];
      var seed = 200;
      using (var file = System.IO.File.Create(@"lorenz_data.csv"))
      using (var writer = new System.IO.StreamWriter(file))
      {
        for (int i = 0; i < N; i++)
        {
          ODE.RungeKutta.ODESolver.Step(x0, y0, f, h, r);
          x0 += h;
          y0[0] = r[0];
          y0[1] = r[1];
          y0[2] = r[2];

          if (i % seed == 0) writer.WriteLine(string.Format("{0},{1}", x0.ToString(System.Globalization.CultureInfo.InvariantCulture), r[0].ToString(System.Globalization.CultureInfo.InvariantCulture)));
        }
      }
    }

    private static void doHeatEquation()
    {
      var x0 = 0.0D;
      var x1 = 1.0D;
      var N  = 500;
      var dx  = (x1 - x0)/N;
      var dx2  = dx*dx;

      var u0 = new double[N+1];
      for (int i=0; i<=N; i++)
      {
        var x = x0 + i*dx;
        u0[i] = System.Math.Exp(-10.0D*(x - 0.5D)*(x - 0.5D));
      }
      var l = (u0[1] + u0[0])/2;
      u0[0] = u0[1] = u0[N-1] = u0[N] = l;

      var buf = new double[N+1];
      var f  = new Func<double, double[], double[]>(
                 (x, y) =>
                 {
                   //var a =(y[0] + y[N-2])/2;
                   //buf[0] = (y[1] - 2*y[0] + a)/dx2;
                   //for (int i=1; i<N-2; i++)
                   //  buf[i] = (y[i+1] - 2*y[i] + y[i-1])/dx2;
                   //buf[N-2] = (a - 2*y[N-2] + y[N-3])/dx2;

                   for (int i=0; i<=N; i++)
                   {
                     var ya = (i==0) ? y[N] : y[i-1];
                     var yb = y[i];
                     var yc = (i==N) ? y[0] : y[i+1];
                     buf[i] = (yc - 2*yb + ya)/dx2 - (1 - ya)*ya;
                   }
                   return buf;
                 });

      var t0 = 0.0D;
      var t1 = 1.0D;
      var dt = 0.0000004D;
      var t = t0;
      var r = new double[N+1];
      using (var file = System.IO.File.Create(@"heat_data.csv"))
      using (var writer = new System.IO.StreamWriter(file))
      {
        while (t < t1)
        {
          ODE.RungeKutta.ODESolver.Step(t, u0, f, dt, r);
          t += dt;
          Array.Copy(r, u0, N-1);
        }

        for (int i=0; i<=N; i++)
        {
          var x = x0 + i*dx;
          double y;
          if (i==0) y=r[0];
          else if (i==N) y=r[N-2];
          else y=r[i-1];

          writer.WriteLine(string.Format("{0},{1}",
                           x.ToString(System.Globalization.CultureInfo.InvariantCulture),
                           y.ToString(System.Globalization.CultureInfo.InvariantCulture)));
        }
      }
    }

    #endregion
  }
}
