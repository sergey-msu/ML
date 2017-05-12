using System;
using ML.Contracts;
using ML.DeepMethods.Models;
using ML.Core.Registry;

namespace ML.Tests.UnitTests.CNN
{
  public static class Mocks
  {
    public interface IMultivarFunction
    {
      double Value(double[][] ww);

      double[][] Gradient(double[][] ww);
    }

    /// <summary>
    /// z = (x+y+z)^2 + 2(x-y+z)^2 + 4(x-z)^2
    /// </summary>
    public class SimpleMultivar : IMultivarFunction
    {
      public double Value(double[][] w)
      {
        var x = w[0][0];
        var y = w[0][1];
        var z = w[1][0];

        return (x+y+z)*(x+y+z) + 2*(x-y+z)*(x-y+z) + 4*(x-z)*(x-z);
      }

      public double[][] Gradient(double[][] w)
      {
        var x = w[0][0];
        var y = w[0][1];
        var z = w[1][0];

        var dfx = 2*(x+y+z) + 4*(x-y+z) + 8*(x-z);
        var dfy = 2*(x+y+z) - 4*(x-y+z);
        var dfz = 2*(x+y+z) + 4*(x-y+z) - 8*(x-z);

        return new double[2][] { new[] { dfx, dfy }, new[] { dfz } };
      }
    }

    /// <summary>
    /// z = x^4 - 10*(x-1)^2 - 5*y*(x+1)^2 + 2*y^4 - 10*(y-1)^2 + 5*(y+1)^2
    ///
    /// MATLAB:
    /// [X,Y] = meshgrid(-4:0.05:3.5,-4:0.05:4);
    /// Z = X.^4-10.*(X-1).^2-5.*(X+1).^2.*Y + 2*Y.^4-10.*(Y-1).^2-5.*(Y+1).^2;
    /// surf(X,Y,Z);
    /// </summary>
    public class MidMultivar : IMultivarFunction
    {
      public double Value(double[][] ww)
      {
        var w1 = ww[0][0];
        var w2 = ww[0][1];

        return Math.Pow(w1, 4) - 10*Math.Pow(w1-1, 2)
               - 5*Math.Pow(w1+1, 2)*w2
               + 2*Math.Pow(w2, 4) - 10*Math.Pow(w2-1, 2) - 5*Math.Pow(w2+1, 2);
      }

      public double[][] Gradient(double[][] ww)
      {
        var w1 = ww[0][0];
        var w2 = ww[0][1];

        var dw1 = 4*Math.Pow(w1, 3) - 20*(w1-1) - 10*w2*(w1+1);
        var dw2 = -5*Math.Pow(w1+1, 2) + 8*Math.Pow(w2, 3) - 30*w2 + 10;
        var res = new[] { dw1, dw2 };

        return new double[1][] { res };
      }
    }

    public class LinearActivation : IActivationFunction
    {
      public string ID { get { return "LA"; } }

      public string Name { get { return "LinearActivation"; } }

      public double Value(double r)
      {
        return (r < 0) ? 2 * r : 3 * r;
      }

      public double Derivative(double r)
      {
        return (r < 0) ? 2 : 3;
      }

      public double DerivativeFromValue(double y)
      {
        return (y < 0) ? 2 : 3;
      }
    }

    public static ConvNet SimpleLinearNetwork(IActivationFunction activation = null)
    {
      activation = activation ?? new Mocks.LinearActivation();

      var net = new ConvNet(1, 1, activation: activation);
      net.IsTraining = true;
      var layer1 = new DenseLayer(1);
      net.AddLayer(layer1);
      var layer2 = new DenseLayer(1);
      net.AddLayer(layer2);
      var layer3 = new DenseLayer(1);
      net.AddLayer(layer3);
      net._Build();

      layer1.Weights[1] = 1;
      layer1.Weights[0] = 3;
      layer2.Weights[1] = -1;
      layer2.Weights[0] = 1;
      layer3.Weights[1] = 2;
      layer3.Weights[0] = -1;

      return net;
    }

    public static ConvNet TestNetwork1()
    {
      var activation = Activation.Atan;
      var net = new ConvNet(1, 2, 2) { IsTraining=true };

      net.AddLayer(new ConvLayer(outputDepth: 2, windowSize: 2, padding: 1, activation: activation));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2));
      net.AddLayer(new ConvLayer(outputDepth: 2, windowSize: 1, activation: activation));
      net.AddLayer(new FlattenLayer(outputDim: 3, activation: activation));
      net.AddLayer(new DenseLayer(outputDim: 2, activation: activation));

      net._Build();

      net.RandomizeParameters(seed: 0);

      return net;
    }


    public static ConvNet SimpleLinearNetworkWithDropout(double drate, int dseed)
    {
      var net = new ConvNet(1, 1, activation: new Mocks.LinearActivation());
      net.IsTraining = true;
      var layer1 = new DenseLayer(1);
      net.AddLayer(layer1);
      var layer2 = new DenseLayer(1);
      net.AddLayer(layer2);
      var layer3 = new DropoutLayer(drate, dseed);
      layer3.Mask = new bool[1][,] { new bool[,] { { true } } };
      net.AddLayer(layer3);
      var layer4 = new DenseLayer(1);
      net.AddLayer(layer4);
      net._Build();

      layer1.Weights[1] =  1;
      layer1.Weights[0] =  3;
      layer2.Weights[1] = -1;
      layer2.Weights[0] =  1;
      layer4.Weights[1] =  2;
      layer4.Weights[0] = -1;

      return net;
    }
  }
}
