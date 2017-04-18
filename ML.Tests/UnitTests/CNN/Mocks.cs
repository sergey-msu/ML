using System;
using ML.Contracts;
using ML.DeepMethods.Models;

namespace ML.Tests.UnitTests.CNN
{
  public static class Mocks
  {
    public class LinearActivation : IActivationFunction
    {
      public string ID { get { return "LA"; } }

      public string Name { get { return "LinearActivation"; } }

      public double Value(double r)
      {
        return (r<0) ? 2*r : 3*r;
      }

      public double Derivative(double r)
      {
        return (r<0) ? 2 : 3;
      }

      public double DerivativeFromValue(double y)
      {
        return (y<0) ? 2 : 3;
      }
    }

    public static ConvNet SimpleLinearNetwork()
    {
      var net = new ConvNet(1, 1, activation: new Mocks.LinearActivation());
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
