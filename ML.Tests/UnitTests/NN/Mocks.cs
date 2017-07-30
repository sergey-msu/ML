using System;
using ML.Contracts;
using ML.NeuralMethods.Models;

namespace ML.Tests.UnitTests.NN
{
  public static class Mocks
  {
    public class LinearActivation : IActivationFunction
    {
      public string Name { get { return "LA"; } }

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

    public static NeuralNetwork SimpleLinearNetwork()
    {
      var net = new NeuralNetwork(1, activation: new Mocks.LinearActivation());
      net.IsTraining = true;
      var layer1 = new NeuralLayer(1);
      net.AddLayer(layer1);
      var layer2 = new NeuralLayer(1);
      net.AddLayer(layer2);
      var layer3 = new NeuralLayer(1);
      net.AddLayer(layer3);
      net.Build();

      layer1[0].Bias = 1;
      layer1[0][0] = 3;
      layer2[0].Bias = -1;
      layer2[0][0] = 1;
      layer3[0].Bias = 2;
      layer3[0][0] = -1;

      return net;
    }

    public static NeuralNetwork SimpleLinearNetworkWithDropout(double drate, double dseed)
    {
      var net = new NeuralNetwork(1, activation: new Mocks.LinearActivation());
      net.IsTraining = true;
      var layer1 = new NeuralLayer(1);
      net.AddLayer(layer1);
      var layer2 = new NeuralLayer(1);
      layer2.DropoutRate = drate;
      layer2.DropoutSeed = 1;
      layer2[0].LastRetained = true;
      net.AddLayer(layer2);
      var layer3 = new NeuralLayer(1);
      net.AddLayer(layer3);
      net.Build();

      layer1[0].Bias = 1;
      layer1[0][0] = 3;
      layer2[0].Bias = -1;
      layer2[0][0] = 1;
      layer3[0].Bias = 2;
      layer3[0][0] = -1;

      return net;
    }
  }
}
