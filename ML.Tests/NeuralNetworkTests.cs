using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ML.Core.Metric;
using ML.Core;
using ML.NeuralMethods;

namespace ML.Tests
{
  [TestClass]
  public class NeuralNetworkTests
  {
    public const double EPS = 0.0000001D;

    #region NeuralNetwork

    [TestMethod]
    public void NeuralNetwork_CreateLayer()
    {
      var net = new NeuralNetwork();

      var layer1 = net.CreateLayer();
      var layer2 = net.CreateLayer();

      Assert.IsNotNull(layer1);
      Assert.IsNotNull(layer2);
      Assert.AreEqual(net, layer1.Network);
      Assert.AreEqual(net, layer2.Network);
      Assert.AreEqual(2, net.Layers.Length);
      Assert.AreEqual(layer1, net[0]);
      Assert.AreEqual(layer2, net[1]);
    }

    [TestMethod]
    public void NeuralNetwork_RemoveLayer()
    {
      var net = new NeuralNetwork();
      var layer1 = net.CreateLayer();
      var layer2 = net.CreateLayer();

      var result = net.RemoveLayer();
      Assert.AreEqual(layer2, result);
      Assert.AreEqual(1, net.Layers.Length);
      Assert.AreEqual(layer1, net[0]);

      result = net.RemoveLayer();
      Assert.AreEqual(layer1, result);
      Assert.AreEqual(0, net.Layers.Length);
    }

    #endregion

    #region NeuralLayer

    [TestMethod]
    public void NeuralLayer_CreateNeuron()
    {
      var net = new NeuralNetwork();
      var layer = net.CreateLayer();

      var neuron1 = layer.CreateNeuron();
      var neuron2 = layer.CreateNeuron();

      Assert.IsNotNull(neuron1);
      Assert.IsNotNull(neuron2);
      Assert.AreEqual(layer, neuron1.Layer);
      Assert.AreEqual(layer, neuron2.Layer);
      Assert.AreEqual(2, layer.Neurons.Length);
      Assert.AreEqual(neuron1, layer[0]);
      Assert.AreEqual(neuron2, layer[1]);
    }

    [TestMethod]
    public void NeuralLayer_RemoveNeuron()
    {
      var net = new NeuralNetwork();
      var layer = net.CreateLayer();

      var neuron1 = layer.CreateNeuron();
      var neuron2 = layer.CreateNeuron();

      var result = layer.RemoveNeuron();
      Assert.AreEqual(neuron2, result);
      Assert.AreEqual(1, layer.Neurons.Length);
      Assert.AreEqual(neuron1, layer[0]);

      result = layer.RemoveNeuron();
      Assert.AreEqual(neuron1, result);
      Assert.AreEqual(0, layer.Neurons.Length);
    }

    [TestMethod]
    public void NeuralLayer_Calculate()
    {
      var net = new NeuralNetwork();
      var layer = net.CreateLayer();
      var neuron1 = layer.CreateNeuron();
      neuron1[0] = 0.1D;
      neuron1[1] = 0.2D;
      var neuron2 = layer.CreateNeuron();
      neuron2[1] = -0.1D;
      neuron2[2] = 0.3D;
      var point = new Point(1, 2, 3);

      layer.ActivationFunction = Registry.ActivationFunctions.Identity;

      var result = layer.Calculate(point);
      Assert.AreEqual(2, result.Length);
      Assert.IsTrue(Math.Abs(result[0] - 0.5) < EPS);
      Assert.IsTrue(Math.Abs(result[1] - 0.7) < EPS);

      layer.ActivationFunction = Registry.ActivationFunctions.Exp;

      result = layer.Calculate(point);
      Assert.AreEqual(2, result.Length);
      Assert.IsTrue(Math.Abs(result[0] - Math.Exp(0.5)) < EPS);
      Assert.IsTrue(Math.Abs(result[1] - Math.Exp(0.7)) < EPS);
    }

    [TestMethod]
    public void NeuralLayer_Calculate_NormOutput()
    {
      var net = new NeuralNetwork();
      var layer = net.CreateLayer();
      layer.NormOutput = true;
      var neuron1 = layer.CreateNeuron();
      neuron1[0] = 0.1D;
      neuron1[1] = 0.2D;
      var neuron2 = layer.CreateNeuron();
      neuron2[1] = -0.1D;
      neuron2[2] = 0.3D;
      var point = new Point(1, 2, 3);

      layer.ActivationFunction = Registry.ActivationFunctions.Identity;

      var result = layer.Calculate(point);
      Assert.AreEqual(2, result.Length);
      Assert.IsTrue(Math.Abs(result[0] - 0.5/1.2) < EPS);
      Assert.IsTrue(Math.Abs(result[1] - 0.7/1.2) < EPS);

      layer.ActivationFunction = Registry.ActivationFunctions.Exp;

      result = layer.Calculate(point);
      Assert.AreEqual(2, result.Length);
      Assert.IsTrue(Math.Abs(result[0] - Math.Exp(0.5)/(Math.Exp(0.5)+Math.Exp(0.7))) < EPS);
      Assert.IsTrue(Math.Abs(result[1] - Math.Exp(0.7)/(Math.Exp(0.5)+Math.Exp(0.7))) < EPS);
    }

    [TestMethod]
    public void NeuralLayer_Calculate_UseBias()
    {
      var net = new NeuralNetwork();
      var layer = net.CreateLayer();
      layer.UseBias = true;
      var neuron1 = layer.CreateNeuron();
      neuron1[0] = 0.1D;
      neuron1[1] = 0.2D;
      neuron1[3] = -0.1D;
      var neuron2 = layer.CreateNeuron();
      neuron2[1] = -0.1D;
      neuron2[2] = 0.3D;
      neuron2[3] = 0.2D;
      var point = new Point(1, 2, 3);

      layer.ActivationFunction = Registry.ActivationFunctions.Identity;

      var result = layer.Calculate(point);
      Assert.AreEqual(2, result.Length);
      Assert.IsTrue(Math.Abs(result[0] - 0.4) < EPS);
      Assert.IsTrue(Math.Abs(result[1] - 0.9) < EPS);

      layer.ActivationFunction = Registry.ActivationFunctions.Exp;

      result = layer.Calculate(point);
      Assert.AreEqual(2, result.Length);
      Assert.IsTrue(Math.Abs(result[0] - Math.Exp(0.4)) < EPS);
      Assert.IsTrue(Math.Abs(result[1] - Math.Exp(0.9)) < EPS);
    }

    [TestMethod]
    public void NeuralLayer_UpdateWeights_Bulk()
    {
      var net = new NeuralNetwork();
      var layer = net.CreateLayer();
      var neuron1 = layer.CreateNeuron();
      neuron1[0] = 0.2D;
      neuron1[1] = -0.1D;
      var neuron2 = layer.CreateNeuron();
      neuron2[1] = -0.1D;
      neuron2[2] = 0.3D;
      var weights = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4 };

      var cursor = 0;
      layer.UpdateWeights(weights, false, ref cursor);
      Assert.AreEqual(4, cursor);
      Assert.AreEqual(1, neuron1[0]);
      Assert.AreEqual(2, neuron1[1]);
      Assert.AreEqual(3, neuron2[1]);
      Assert.AreEqual(4, neuron2[2]);

      layer.UpdateWeights(weights, false, ref cursor);
      Assert.AreEqual(8, cursor);
      Assert.AreEqual(5, neuron1[0]);
      Assert.AreEqual(6, neuron1[1]);
      Assert.AreEqual(7, neuron2[1]);
      Assert.AreEqual(8, neuron2[2]);

      layer.UpdateWeights(weights, false, ref cursor);
      Assert.AreEqual(12, cursor);
      Assert.AreEqual(9, neuron1[0]);
      Assert.AreEqual(0, neuron1[1]);
      Assert.AreEqual(1, neuron2[1]);
      Assert.AreEqual(2, neuron2[2]);

      layer.UpdateWeights(weights, false, ref cursor);
      Assert.AreEqual(14, cursor);
      Assert.AreEqual(3, neuron1[0]);
      Assert.AreEqual(4, neuron1[1]);
      Assert.AreEqual(1, neuron2[1]);
      Assert.AreEqual(2, neuron2[2]);
    }

    [TestMethod]
    public void NeuralLayer_UpdateWeights_Delta()
    {
      var net = new NeuralNetwork();
      var layer = net.CreateLayer();
      var neuron1 = layer.CreateNeuron();
      neuron1[0] = 2D;
      neuron1[1] = -1D;
      var neuron2 = layer.CreateNeuron();
      neuron2[1] = -1D;
      neuron2[2] = 3D;
      var weights = new double[] { 1, 2, -1, 1, 1, 2, 1, 0, 1, 0, -1, 2, 1, -1 };

      var cursor = 0;
      layer.UpdateWeights(weights, true, ref cursor);
      Assert.AreEqual(4, cursor);
      Assert.AreEqual(3, neuron1[0]);
      Assert.AreEqual(1, neuron1[1]);
      Assert.AreEqual(-2, neuron2[1]);
      Assert.AreEqual(4, neuron2[2]);

      layer.UpdateWeights(weights, true, ref cursor);
      Assert.AreEqual(8, cursor);
      Assert.AreEqual(4, neuron1[0]);
      Assert.AreEqual(3, neuron1[1]);
      Assert.AreEqual(-1, neuron2[1]);
      Assert.AreEqual(4, neuron2[2]);

      layer.UpdateWeights(weights, true, ref cursor);
      Assert.AreEqual(12, cursor);
      Assert.AreEqual(5, neuron1[0]);
      Assert.AreEqual(3, neuron1[1]);
      Assert.AreEqual(-2, neuron2[1]);
      Assert.AreEqual(6, neuron2[2]);

      layer.UpdateWeights(weights, true, ref cursor);
      Assert.AreEqual(14, cursor);
      Assert.AreEqual(6, neuron1[0]);
      Assert.AreEqual(2, neuron1[1]);
      Assert.AreEqual(-2, neuron2[1]);
      Assert.AreEqual(6, neuron2[2]);
    }

    #endregion

    #region Neuron

    [TestMethod]
    public void Neuron_SetWeights()
    {
      var net = new NeuralNetwork();
      var layer = net.CreateLayer();
      var neuron = layer.CreateNeuron();

      neuron[0] = 0.1D;
      neuron[2] = 0.2D;
      neuron[3] = -0.1D;

      Assert.AreEqual(3, neuron.WeightCount);
      Assert.AreEqual(0.1D,  neuron[0]);
      Assert.AreEqual(null,  neuron[1]);
      Assert.AreEqual(0.2D,  neuron[2]);
      Assert.AreEqual(-0.1D, neuron[3]);
      Assert.AreEqual(null,  neuron[4]);
    }

    [TestMethod]
    public void Neuron_Calculate()
    {
      var net = new NeuralNetwork();
      var layer = net.CreateLayer();
      var neuron = layer.CreateNeuron();
      var point = new Point(1, 2, 3, 4);

      neuron[0] = 0.1D;
      neuron[2] = 0.2D;
      neuron[3] = -0.1D;

      neuron.ActivationFunction = Registry.ActivationFunctions.Identity;
      var result = neuron.Calculate(point);
      Assert.IsTrue(Math.Abs(result - 0.3) < EPS);

      neuron.ActivationFunction = Registry.ActivationFunctions.Exp;
      result = neuron.Calculate(point);
      Assert.IsTrue(Math.Abs(result - Math.Exp(0.3)) < EPS);
    }

    [TestMethod]
    public void Neuron_UpdateWeights_Bulk()
    {
      var net = new NeuralNetwork();
      var layer = net.CreateLayer();
      var neuron = layer.CreateNeuron();
      var weights = new double[] { 1, 2, 3, 4, 5, 6, 7, 8 };

      neuron[0] = 0.1D;
      neuron[2] = 0.2D;
      neuron[3] = -0.1D;

      var cursor = 0;
      neuron.UpdateWeights(weights, false, ref cursor);
      Assert.AreEqual(3,    cursor);
      Assert.AreEqual(1,    neuron[0]);
      Assert.AreEqual(null, neuron[1]);
      Assert.AreEqual(2,    neuron[2]);
      Assert.AreEqual(3,    neuron[3]);

      neuron.UpdateWeights(weights, false, ref cursor);
      Assert.AreEqual(6,    cursor);
      Assert.AreEqual(4,    neuron[0]);
      Assert.AreEqual(null, neuron[1]);
      Assert.AreEqual(5,    neuron[2]);
      Assert.AreEqual(6,    neuron[3]);

      neuron.UpdateWeights(weights, false, ref cursor);
      Assert.AreEqual(8,    cursor);
      Assert.AreEqual(7,    neuron[0]);
      Assert.AreEqual(null, neuron[1]);
      Assert.AreEqual(8,    neuron[2]);
      Assert.AreEqual(6,    neuron[3]);
    }

    [TestMethod]
    public void Neuron_UpdateWeights_Delta()
    {
      var net = new NeuralNetwork();
      var layer = net.CreateLayer();
      var neuron = layer.CreateNeuron();
      var weights = new double[] { 1, 2, 0, 1, -1, 2, 1, 0 };

      neuron[0] = 1;
      neuron[2] = 2;
      neuron[3] = 1;

      var cursor = 0;
      neuron.UpdateWeights(weights, true, ref cursor);
      Assert.AreEqual(3,    cursor);
      Assert.AreEqual(2,    neuron[0]);
      Assert.AreEqual(null, neuron[1]);
      Assert.AreEqual(4,    neuron[2]);
      Assert.AreEqual(1,    neuron[3]);

      neuron.UpdateWeights(weights, true, ref cursor);
      Assert.AreEqual(6,    cursor);
      Assert.AreEqual(3,    neuron[0]);
      Assert.AreEqual(null, neuron[1]);
      Assert.AreEqual(3,    neuron[2]);
      Assert.AreEqual(3,    neuron[3]);

      neuron.UpdateWeights(weights, true, ref cursor);
      Assert.AreEqual(8,    cursor);
      Assert.AreEqual(4,    neuron[0]);
      Assert.AreEqual(null, neuron[1]);
      Assert.AreEqual(3,    neuron[2]);
      Assert.AreEqual(3,    neuron[3]);
    }


    #endregion
  }
}
