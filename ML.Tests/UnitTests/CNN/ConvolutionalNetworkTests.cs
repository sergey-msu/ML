using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ML.Core;
using ML.DeepMethods.Models;
using ML.Core.Registry;

namespace ML.Tests.UnitTests.CNN
{
  [TestClass]
  public class ConvolutionalNetworkTests : TestBase
  {
    [ClassInitialize]
    public static void ClassInit(TestContext context)
    {
      BaseClassInit(context);
    }

    #region ConvLayer

    [TestMethod]
    public void ConvLayer_Calculate_Simple()
    {
      var layer = getSimpleConvolutionLayer();
      layer.Build();

      var input = new double[2][,]
      {
        // channel #1
        new double[3,3]
        {
          { 1, 0, 1 },
          { 0, 1, 1 },
          { 0, 0, 1}
        },

        // channel #2
        new double[3,3]
        {
          { 1, 0, 0 },
          { 1, 0, 0 },
          { 0, 1, 0 }
        }
      };

      var result = layer.Calculate(input);

      Assert.AreEqual(3, result.GetLength(0));

      //fm #1
      Assert.AreEqual(2, layer.Value(0, 0, 0));
      Assert.AreEqual(1, layer.Value(0, 0, 1));
      Assert.AreEqual(1, layer.Value(0, 1, 0));
      Assert.AreEqual(2, layer.Value(0, 1, 1));

      //fm #2
      Assert.AreEqual(5, layer.Value(1, 0, 0));
      Assert.AreEqual(4, layer.Value(1, 0, 1));
      Assert.AreEqual(4, layer.Value(1, 1, 0));
      Assert.AreEqual(6, layer.Value(1, 1, 1));

      //fm #3
      Assert.AreEqual(0, layer.Value(2, 0, 0));
      Assert.AreEqual(0, layer.Value(2, 0, 1));
      Assert.AreEqual(2, layer.Value(2, 1, 0));
      Assert.AreEqual(0, layer.Value(2, 1, 1));
    }

    [TestMethod]
    public void ConvLayer_Calculate_Medium()
    {
      var layer = getMediumConvolutionLayer();
      layer.Build();

      var input = new double[2][, ]
      {
        // channel #1
        new double[4,4]
        {
          { 1, 0, 1, 0 },
          { 0, 1, 1, 0 },
          { 0, 0, 1, 0 },
          { 1, 0, 0, 0 },
        },

        // channel #2
        new double[4,4]
        {
          { 1, 0, 0, 0 },
          { 1, 1, 0, 0 },
          { 0, 0, 0, 1 },
          { 1, 1, 0, 1 },
        }
      };

      var result = layer.Calculate(input);

      Assert.AreEqual(2, result.GetLength(0));

      //fm #1
      Assert.AreEqual(6, layer.Value(0, 0, 0));
      Assert.AreEqual(2, layer.Value(0, 0, 1));
      Assert.AreEqual(2, layer.Value(0, 1, 0));
      Assert.AreEqual(5, layer.Value(0, 1, 1));

      //fm #2
      Assert.AreEqual(0, layer.Value(1, 0, 0));
      Assert.AreEqual(0, layer.Value(1, 0, 1));
      Assert.AreEqual(0, layer.Value(1, 1, 0));
      Assert.AreEqual(2, layer.Value(1, 1, 1));
    }

    #endregion

    #region MaxPoolingLayer

    [TestMethod]
    public void MaxPoolingLayer_Calculate_Simple()
    {
      var layer = getSimplePoolingLayer<MaxPoolingLayer>();

      var input = new double[2][,]
      {
        // channel #1
        new double[3,3]
        {
          { 2, 0, 1 },
          { 0, 1, 1 },
          { 0, 0, 1}
        },

        // channel #2
        new double[3,3]
        {
          { 1, 0, 0 },
          { 1, 0, 0 },
          { 0, 3, 0 }
        }
      };

      var result = layer.Calculate(input);

      Assert.AreEqual(2, result.GetLength(0));
      Assert.AreEqual(2, result.GetLength(1));
      Assert.AreEqual(2, result.GetLength(2));

      //fm #1
      Assert.AreEqual(2, result[0][0, 0]);
      Assert.AreEqual(1, result[0][0, 1]);
      Assert.AreEqual(1, result[0][1, 0]);
      Assert.AreEqual(1, result[0][1, 1]);

      //fm #2
      Assert.AreEqual(1, result[1][0, 0]);
      Assert.AreEqual(0, result[1][0, 1]);
      Assert.AreEqual(3, result[1][1, 0]);
      Assert.AreEqual(3, result[1][1, 1]);
    }

    [TestMethod]
    public void MaxPoolingLayer_Calculate_Medium()
    {
      var layer = getMediumPoolingLayer<MaxPoolingLayer>();
      layer.Build();

      var input = new double[2][,]
      {
        // channel #1
        new double[4,4]
        {
          { 1, 0, 1, 0 },
          { 0, 3, 1, 0 },
          { 0, 0, 1, 0 },
          { 1, 0, 0, 5 },
        },

        // channel #2
        new double[4,4]
        {
          { 1, 0, 0,-2 },
          { 1, 1, 0, 0 },
          { 0, 0, 0, 1 },
          { 4, 1, 0, 3 },
        }
      };

      var result = layer.Calculate(input);

      Assert.AreEqual(2, result.GetLength(0));

      //fm #1
      Assert.AreEqual(3, result[0][0, 0]);
      Assert.AreEqual(3, result[0][0, 1]);
      Assert.AreEqual(3, result[0][1, 0]);
      Assert.AreEqual(5, result[0][1, 1]);

      //fm #2
      Assert.AreEqual(1, result[1][0, 0]);
      Assert.AreEqual(1, result[1][0, 1]);
      Assert.AreEqual(4, result[1][1, 0]);
      Assert.AreEqual(3, result[1][1, 1]);
    }

    #endregion

    #region AvgPoolingLayer

    [TestMethod]
    public void AvgPoolingLayer_Calculate_Simple()
    {
      var layer = getSimplePoolingLayer<AvgPoolingLayer>();

      var input = new double[2][,]
      {
        // channel #1
        new double[3,3]
        {
          { 2, 0, 1 },
          { 0, 1, 1 },
          { 0, 0, 1}
        },

        // channel #2
        new double[3,3]
        {
          { 1, 0, 0 },
          { 1, 0, 0 },
          { 0, 3, 0 }
        }
      };

      var result = layer.Calculate(input);

      Assert.AreEqual(2, result.GetLength(0));
      Assert.AreEqual(2, result.GetLength(1));
      Assert.AreEqual(2, result.GetLength(2));

      //fm #1
      Assert.AreEqual(0.75D, result[0][0, 0]);
      Assert.AreEqual(0.75D, result[0][0, 1]);
      Assert.AreEqual(0.25D, result[0][1, 0]);
      Assert.AreEqual(0.75D, result[0][1, 1]);

      //fm #2
      Assert.AreEqual(0.5D,  result[1][0, 0]);
      Assert.AreEqual(0.0D,  result[1][0, 1]);
      Assert.AreEqual(1.0D,  result[1][1, 0]);
      Assert.AreEqual(0.75D, result[1][1, 1]);
    }

    [TestMethod]
    public void AvgPoolingLayer_Calculate_Medium()
    {
      var layer = getMediumPoolingLayer<AvgPoolingLayer>();
      layer.Build();

      var input = new double[2][,]
      {
        // channel #1
        new double[4,4]
        {
          { 1, 0, 1, 0 },
          { 0, 3, 1, 0 },
          { 0, 0, 1, 0 },
          { 1, 0, 0, 5 },
        },

        // channel #2
        new double[4,4]
        {
          { 1, 0, 0,-3 },
          { 1, 1, 0, 0 },
          { 0, 0, 0, 1 },
          { 4, 1, 0, 3 },
        }
      };

      var result = layer.Calculate(input);

      Assert.AreEqual(2, result.GetLength(0));

      //fm #1
      Assert.AreEqual(7.0D/16,  result[0][0, 0], EPS);
      Assert.AreEqual(6.0D/16,  result[0][0, 1], EPS);
      Assert.AreEqual(6.0D/16,  result[0][1, 0]);
      Assert.AreEqual(10.0D/16, result[0][1, 1]);

      //fm #2
      Assert.AreEqual( 3.0D/16, result[1][0, 0]);
      Assert.AreEqual(-1.0D/16, result[1][0, 1]);
      Assert.AreEqual( 7.0D/16, result[1][1, 0]);
      Assert.AreEqual( 6.0D/16, result[1][1, 1]);
    }

    #endregion

    #region ConvolutionalNetwork

    [TestMethod]
    public void ConvolutionalNetwork_Calculate()
    {
      var net = getConvolutionalNetwork();
      net.Build();

      var input = new double[1][,]
      {
        new double[8,8]
        {
          { 0, 0, 0, 0, 1, 0, 0, 0 },
          { 0, 0, 0, 1, 1, 0, 0, 0 },
          { 0, 0, 1, 1, 1, 0, 0, 0 },
          { 0, 0, 0, 1, 1, 0, 0, 0 },
          { 0, 0, 0, 1, 1, 0, 0, 0 },
          { 0, 0, 0, 1, 1, 0, 0, 0 },
          { 0, 0, 1, 1, 1, 1, 0, 0 },
          { 0, 0, 1, 1, 1, 1, 0, 0 }
        }
      };

      var result = net.Calculate(input);

      Assert.AreEqual(1, net[0].Value(0, 0, 0));
      Assert.AreEqual(2, net[0].Value(0, 1, 1));
      Assert.AreEqual(5, net[0].Value(0, 2, 2));
      Assert.AreEqual(1, net[0].Value(0, 3, 3));
      Assert.AreEqual(0, net[0].Value(1, 0, 0));
      Assert.AreEqual(2, net[0].Value(1, 1, 1));
      Assert.AreEqual(1, net[0].Value(1, 2, 2));
      Assert.AreEqual(0, net[0].Value(1, 3, 3));
      Assert.AreEqual(2, net[0].Value(2, 0, 0));
      Assert.AreEqual(4, net[0].Value(2, 1, 1));
      Assert.AreEqual(3, net[0].Value(2, 2, 2));
      Assert.AreEqual(1, net[0].Value(2, 3, 3));
      Assert.AreEqual(1, net[0].Value(3, 0, 0));
      Assert.AreEqual(3, net[0].Value(3, 1, 1));
      Assert.AreEqual(3, net[0].Value(3, 2, 2));
      Assert.AreEqual(1, net[0].Value(3, 3, 3));

      Assert.AreEqual(2, net[1].Value(0, 0, 0));
      Assert.AreEqual(5, net[1].Value(0, 1, 1));
      Assert.AreEqual(2, net[1].Value(1, 0, 0));
      Assert.AreEqual(2, net[1].Value(1, 1, 1));
      Assert.AreEqual(4, net[1].Value(2, 0, 0));
      Assert.AreEqual(3, net[1].Value(2, 0, 1));
      Assert.AreEqual(3, net[1].Value(3, 0, 0));
      Assert.AreEqual(3, net[1].Value(3, 1, 1));

      Assert.AreEqual(0,   net[2].Value(0, 0, 0));
      Assert.AreEqual(100, net[2].Value(2, 0, 0));
      Assert.AreEqual(200, net[2].Value(4, 0, 0));
      Assert.AreEqual(350, net[2].Value(7, 0, 0));

      Assert.AreEqual(2, result.GetLength(0));
      Assert.AreEqual(2, result.Length);
      Assert.AreEqual( 1401, result[0][0,0]);
      Assert.AreEqual(-1401, result[1][0,0]);
    }

    #endregion

    #region .pvt

    private ConvLayer getSimpleConvolutionLayer()
    {
      const int inputDepth  = 2;
      const int outputDepth = 3;
      const int windowSize  = 2;

      var layer = new ConvLayer(outputDepth, windowSize, stride:1, padding:1) { IsTraining=true };
      layer.ActivationFunction = Activation.ReLU;

      const int plen = (windowSize*windowSize*inputDepth + 1)*outputDepth;
      var kernel = new double[plen]
                   {
                     // fm #1
                       // ch #1
                       1, 0,
                       1,-1,
                       // ch #2
                       0, 2,
                       1, 0,
                     1, // bias #1

                     // fm #2
                       // ch #1
                       1, 1,
                       0, 0,
                       // ch #2
                       0, 1,
                       1, 0,
                     3, // bias #2

                     // fm #3
                       // ch #1
                      -2, 0,
                       0,-1,
                       // ch #2
                       1, 0,
                       1, 0,
                     1 // bias #3
                   };
      int cursor = 0;
      layer.TryUpdateParams(kernel, false, ref cursor);

      return layer;
    }

    private ConvLayer getMediumConvolutionLayer()
    {
      const int inputDepth  = 2;
      const int outputDepth = 2;
      const int windowSize  = 4;

      var layer = new ConvLayer(outputDepth, windowSize, stride:2, padding:1) { IsTraining=true };
      layer.ActivationFunction = Activation.ReLU;

      const int plen = (windowSize*windowSize*inputDepth + 1)*outputDepth;
      var kernel = new double[plen]
                   {
                     // fm #1
                       // ch #1
                       1, 1, 0, 0,
                       1, 1, 0, 0,
                       0, 0, 1, 1,
                       0, 0, 1, 1,
                       // ch #2
                       0, 0, 0, 1,
                       0, 0, 1, 0,
                       0, 1, 0, 0,
                       1, 0, 0, 0,
                     1, // bias #1

                     // fm #2
                       // ch #1
                       2, 0, 0, 1,
                       0,-1, 1, 0,
                       0, 1,-1, 0,
                       1, 0, 0, 3,
                       // ch #2
                       2, 0, 0, 1,
                      -3, 0, 1, 0,
                       0, 0, 0, 0,
                       0, 0, 0,-4,
                     -2
                   };
      int cursor = 0;
      layer.TryUpdateParams(kernel, false, ref cursor);

      return layer;
    }

    private TPool getSimplePoolingLayer<TPool>()
      where TPool : PoolingLayer
    {
      const int inputDepth  = 2;
      const int inputSize   = 3;
      const int windowSize  = 2;
      const int stride  = 1;
      const int padding = 0;

      var layer = (TPool)Activator.CreateInstance(typeof(TPool), windowSize, stride, padding, true);

      return layer;
    }

    private TPool getMediumPoolingLayer<TPool>()
      where TPool : PoolingLayer
    {
      const int inputDepth  = 2;
      const int inputSize   = 4;
      const int windowSize  = 4;
      const int stride  = 2;
      const int padding = 1;

      var layer = (TPool)Activator.CreateInstance(typeof(TPool), windowSize, stride, padding, true);

      return layer;
    }

    private ConvNet getConvolutionalNetwork()
    {
      var net = new ConvNet(1, 8);

      // First layer: convolutional layer

      var conv = new ConvLayer(outputDepth: 4,
                                        windowSize: 4,
                                        stride: 2,
                                        padding: 1) { IsTraining=true };
      conv.ActivationFunction = Activation.ReLU;
      var kernel = new double[(4*4*1+1)*4]
                   {
                     // fm #1
                       // ch #1
                       0, 1, 0, 0,
                       0, 1, 0, 0,
                       0, 1, 0, 0,
                       0, 1, 0, 0,
                     1, // bias #1

                     // fm #2
                       // ch #1
                       0, 0, 0, 0,
                       1, 1, 1, 1,
                       0, 0, 0, 0,
                       0, 0, 0, 0,
                     -1, // bias #2

                     // fm #3
                       // ch #1
                       1, 0, 0, 0,
                       0, 1, 0, 0,
                       0, 0, 1, 0,
                       0, 0, 0, 1,
                     1, // bias #3

                     // fm #4
                       // ch #1
                       0, 0, 0, 1,
                       0, 0, 1, 0,
                       0, 1, 0, 0,
                       1, 0, 0, 0,
                     1 // bias #3
                   };
      int cursor = 0;
      conv.TryUpdateParams(kernel, false, ref cursor);
      net.AddLayer(conv);

      // Second layer: max pooling
      var mp = new MaxPoolingLayer(windowSize: 2, stride: 2);
      net.AddLayer(mp);

      // Third layer: fully-connected layer
      var fc = new ConvLayer(outputDepth: 8,
                                      windowSize: 2,
                                      stride: 1) { IsTraining=true };
      fc.ActivationFunction = Activation.Identity;
      var pcount = (2*2*4+1)*8;
      kernel = new double[pcount];
      for (int i=0; i<pcount; i++)
        kernel[i] = i / (2*2*4+1);

      cursor = 0;
      fc.TryUpdateParams(kernel, false, ref cursor);
      net.AddLayer(fc);

      // Fourth layer: output
      var output = new ConvLayer(outputDepth: 2,
                                          windowSize: 1,
                                          stride: 1) { IsTraining=true };
      output.ActivationFunction = Activation.Identity;
      pcount = (1*1*8+1)*2;
      kernel = new double[pcount];
      for (int i=0; i<pcount; i++)
        kernel[i] = (i<=(1*1*8+1)) ? 1 : -1;
      cursor = 0;
      output.TryUpdateParams(kernel, false, ref cursor);

      net.AddLayer(output);

      return net;
    }

    #endregion
  }
}
