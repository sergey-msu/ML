using System;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ML.Core.ComputingNetworks;
using ML.Core;

namespace ML.Tests.UnitTests.Core
{
  [TestClass]
  public class ComputingNodeTests : TestBase
  {
    #region Inner

    #region ComputingNodes

    public class ScalarProductNode : ComputingNode<double[], double>
    {
      private double[] m_Coeffs;

      public ScalarProductNode(params double[] coeffs)
      {
        m_Coeffs = coeffs;
      }

      public double[] Coeffs { get { return m_Coeffs; } }

      public override int ParamCount { get { return m_Coeffs.Length; } }

      public override double Calculate(double[] input)
      {
        var result = 0.0D;
        for (int i=0; i< m_Coeffs.Length; i++)
          result += m_Coeffs[i]*input[i];

        return result;
      }

      protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor)
      {
        for (int i=0; i<m_Coeffs.Length; i++)
        {
          if (cursor+i >= pars.Length) break;
          if (isDelta)
            m_Coeffs[i] += pars[cursor+i];
          else
            m_Coeffs[i] = pars[cursor+i];
        }
      }

      protected override double DoGetParam(int idx)
      {
        return m_Coeffs[idx];
      }

      protected override void DoSetParam(int idx, double value, bool isDelta)
      {
        if (isDelta)
          m_Coeffs[idx] += value;
        else
          m_Coeffs[idx] = value;
      }
    }

    public class DoublingNode : ComputingNode<double, double>
    {
      public override int ParamCount { get { return 0; } }

      public override double Calculate(double input)
      {
        return input * 2;
      }

      protected override double DoGetParam(int idx) { return 0; }
      protected override void DoSetParam(int idx, double value, bool isDelta) { }
      protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor) { }
    }

    public class IdentityNode : ComputingNode<double, double>
    {
      public override int ParamCount { get { return 0; } }

      public override double Calculate(double input)
      {
        return input;
      }

      protected override double DoGetParam(int idx) { return 0; }
      protected override void DoSetParam(int idx, double value, bool isDelta) { }
      protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor) { }
    }

    public class ShiftingNode : ComputingNode<double, double>
    {
      public double m_Shift;

      public ShiftingNode(double shift)
      {
        m_Shift = shift;
      }

      public override int ParamCount { get { return 1; } }

      public double Shift { get { return m_Shift; } }

      public override double Calculate(double input)
      {
        return input + m_Shift;
      }

      protected override double DoGetParam(int idx)
      {
        return m_Shift;
      }

      protected override void DoSetParam(int idx, double value, bool isDelta)
      {
        if (isDelta)
          m_Shift += value;
        else
          m_Shift = value;
      }

      protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor)
      {
        if (isDelta)
          m_Shift += pars[cursor];
        else
          m_Shift = pars[cursor];
      }
    }

    public class PowerNode : ComputingNode<double, double>
    {
      public double m_Power;

      public PowerNode(double power)
      {
        m_Power = power;
      }

      public double Power { get { return m_Power; } }

      public override int ParamCount { get { return 1; } }

      public override double Calculate(double input)
      {
        return Math.Pow(Math.Abs(input), m_Power);
      }

      protected override double DoGetParam(int idx)
      {
        return m_Power;
      }

      protected override void DoSetParam(int idx, double value, bool isDelta)
      {
        if (isDelta)
          m_Power += value;
        else
          m_Power = value;
      }

      protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor)
      {
        if (isDelta)
          m_Power += pars[cursor];
        else
          m_Power = pars[cursor];
      }
    }

    public class PolynomialNode : ComputingNode<double, double>
    {
      private int m_Degree;
      private double[] m_Coeffs;

      public PolynomialNode(int degree, double[] coeffs)
      {
        m_Degree = degree;
        m_Coeffs = coeffs;
      }

      public int Degree { get { return m_Degree; } }
      public double[] Coeffs { get { return m_Coeffs; } }

      public override int ParamCount { get { return m_Coeffs.Length+1; } }

      public override double Calculate(double input)
      {
        var result = m_Coeffs[m_Degree];
        for (int i=m_Degree; i>0; i--)
          result = result*input + m_Coeffs[i-1];

        return result;
      }

      protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor)
      {
        if (isDelta)
          m_Degree += (int)pars[cursor];
        else
          m_Degree = (int)pars[cursor];

        for (int i=0; i<m_Coeffs.Length; i++)
        {
          if (cursor+i+1 >= pars.Length) break;
          if (isDelta)
            m_Coeffs[i] += pars[cursor+i+1];
          else
            m_Coeffs[i] = pars[cursor+i+1];
        }
      }

      protected override double DoGetParam(int idx)
      {
        if (idx==0) return m_Degree;
        return m_Coeffs[idx-1];
      }

      protected override void DoSetParam(int idx, double value, bool isDelta)
      {
        if (idx==0)
        {
          if (isDelta)
            m_Degree += (int)value;
          else
            m_Degree = (int)value;
          return;
        }

        if (isDelta)
          m_Coeffs[idx-1] += value;
        else
          m_Coeffs[idx-1] = value;
      }
    }

    public class MatrixNode : ComputingNode<double[], double[]>
    {
      public double[] m_Coeffs;

      public MatrixNode(double a11, double a12, double a21, double a22)
      {
        m_Coeffs = new[] { a11, a12, a21, a22 };
      }

      public double[] Coeffs { get { return m_Coeffs; } }

      public override int ParamCount { get { return m_Coeffs.Length; } }

      public override double[] Calculate(double[] input)
      {
        var y1 = m_Coeffs[0]*input[0] + m_Coeffs[1]*input[1];
        var y2 = m_Coeffs[1]*input[0] + m_Coeffs[3]*input[1];

        return new[] { y1, y2 };
      }

      protected override double DoGetParam(int idx)
      {
        return m_Coeffs[idx];
      }

      protected override void DoSetParam(int idx, double value, bool isDelta)
      {
        if (isDelta)
          m_Coeffs[idx] += value;
        else
          m_Coeffs[idx] = value;
      }

      protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor)
      {
        for (int i=0; i<m_Coeffs.Length; i++)
        {
          if (cursor+i >= pars.Length) break;
          if (isDelta)
            m_Coeffs[i] += pars[cursor+i];
          else
            m_Coeffs[i] = pars[cursor+i];
        }
      }
    }

    public class DoubleStepNode : ComputingNode<double, double>
    {
      public override int ParamCount { get { return 0; } }

      public override double Calculate(double input)
      {
        return 2 * (int)(input / 2);
      }

      protected override double DoGetParam(int idx) { return 0; }
      protected override void DoSetParam(int idx, double value, bool isDelta) { }
      protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor) { }
    }

    public class MaxNode : ComputingNode<Point2D, double>
    {
      public override int ParamCount { get { return 0; } }

      public override double Calculate(Point2D input)
      {
        return Math.Max(input.X, input.Y);
      }

      protected override double DoGetParam(int idx) { return 0; }
      protected override void DoSetParam(int idx, double value, bool isDelta) { }
      protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor) { }
    }

    public class NMatrixNode : ComputingNode<double[], double[]>
    {
      private double[,] m_Coeffs;

      public NMatrixNode(double[,] coeffs)
      {
        m_Coeffs = new double[coeffs.GetLength(0), coeffs.GetLength(1)];
        Array.Copy(coeffs, m_Coeffs, coeffs.Length);
      }

      public override int ParamCount { get { return m_Coeffs.Length; } }

      public override double[] Calculate(double[] input)
      {
        var dim = m_Coeffs.GetLength(0);
        var result = new double[dim];

        for (int i=0; i<dim; i++)
        {
          var sum = 0.0D;
          for (int j=0; j<input.Length; j++)
            sum += m_Coeffs[i,j]*input[j];
          result[i] = sum;
        }

        return result;
      }

      protected override double DoGetParam(int idx)
      {
        var i = idx / m_Coeffs.GetLength(0);
        var j = idx % m_Coeffs.GetLength(0);
        return m_Coeffs[i,j];
      }

      protected override void DoSetParam(int idx, double value, bool isDelta)
      {
        var i = idx / m_Coeffs.GetLength(0);
        var j = idx % m_Coeffs.GetLength(0);
        if (isDelta)
          m_Coeffs[i,j] += value;
        else
          m_Coeffs[i,j] = value;
      }

      protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor)
      {
        for (int l=0; l<m_Coeffs.Length; l++)
        {
          if (cursor+l >= pars.Length) break;
          var i = l / m_Coeffs.GetLength(0);
          var j = l % m_Coeffs.GetLength(0);
          if (isDelta)
            m_Coeffs[i,j] += pars[cursor+l];
          else
            m_Coeffs[i,j] = pars[cursor+l];
        }
      }
    }

    public class MergeNode : ComputingNode<double[], double>
    {
      public double[] m_Coeffs;

      public MergeNode(double w1, double w2)
      {
        m_Coeffs = new[] { w1, w2 };
      }

      public double[] Coeffs { get { return m_Coeffs; } }

      public override int ParamCount { get { return m_Coeffs.Length; } }

      protected override double DoGetParam(int idx)
      {
        return m_Coeffs[idx];
      }

      protected override void DoSetParam(int idx, double value, bool isDelta)
      {
        if (isDelta)
          m_Coeffs[idx] += value;
        else
          m_Coeffs[idx] = value;
      }

      protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor)
      {
        for (int i=0; i<m_Coeffs.Length; i++)
        {
          if (cursor+i >= pars.Length) break;
          if (isDelta)
            m_Coeffs[i] += pars[cursor+i];
          else
            m_Coeffs[i] = pars[cursor+i];
        }
      }

      public override double Calculate(double[] results)
      {
        return m_Coeffs[0]*results[0] + m_Coeffs[1]*results[1];
      }
    }

    #endregion

    #region Computing Networks

    /// <summary>
    ///                       DoubleStep(x -> 2*(int)(x/2)) -> Shift(x -> x+w3)
    ///                      /                                                 \
    ///  (x,y) -> max(x,y) ->                                                   Sc.pr.(w1, w2) -> Shift(x -> x+w5)
    ///                      \                                                 /
    ///                                       Pow(x -> x^w4)
    ///  params: [w1, w2, w3, w4, w5]
    /// </summary>
    public class SmallNetwork : JoinNode<Point2D, double, double>
    {
      public SmallNetwork()
      {
        this.SetInputNode(new MaxNode());

        var seq = new SequenceNode<double>();

        var hidden = new AggregateNode<double, double>();
        var upper = new SequenceNode<double>();
        upper.AddSubNode(new DoubleStepNode());
        upper.AddSubNode(new ShiftingNode(-1));
        hidden.AddSubNode(upper);
        hidden.AddSubNode(new PowerNode(2));
        hidden.SetMergeNode(new MergeNode(2, -1));
        seq.AddSubNode(hidden);
        seq.AddSubNode(new ShiftingNode(3));

        this.SetOutputNode(seq);
      }
    }

    /// <summary>
    /// (x1,...,x10) -> 10 of 10x10 matrix transform -> scalar product
    ///
    /// params: [w[1,1,1], ..., w[10,10,1], ..., w[10,10,10], w1, w2, ..., w10] = #1010
    /// </summary>
    public class LargeNetwork : JoinNode<double[], double[], double>
    {
      public LargeNetwork(int dim, int lcount)
      {
        var M = new double[dim, dim];
        for (int i=0; i<dim; i++)
        for (int j=0; j<dim; j++)
          M[i,j] = (double)(i+j)/dim;

        var seq = new SequenceNode<double[]>();
        for (int l=0; l<lcount; l++)
        {
          var node = new NMatrixNode(M);
          seq.AddSubNode(node);
        }

        var coeffs = new double[dim];
        for (int i=1; i<dim; i++)
          coeffs[i] = 1;

        this.SetInputNode(seq);
        this.SetOutputNode(new ScalarProductNode(coeffs));
      }
    }

    #endregion

    #endregion

    [ClassInitialize]
    public static void ClassInit(TestContext context)
    {
      BaseClassInit(context);
    }

    #region ComputingNode

    [TestMethod]
    public void ComputingNode_Build()
    {
      var node = new ScalarProductNode(2, -1);
      node.Build();

      Assert.AreEqual(2, node.ParamCount);
    }

    [TestMethod]
    public void ComputingNode_Calculate()
    {
      var node = new ScalarProductNode(2, -1);
      node.Build();

      var input = new double[] { 3, 4 };
      var res = node.Calculate(input);

      Assert.AreEqual(2, res);
    }

    [TestMethod]
    public void ComputingNode_TryGetParam()
    {
      var node = new ScalarProductNode(2, -1);
      node.Build();

      double par1;
      double par2;
      double par3;
      var res1 = node.TryGetParam(0, out par1);
      var res2 = node.TryGetParam(1, out par2);
      var res3 = node.TryGetParam(2, out par3);

      Assert.IsTrue(res1);
      Assert.AreEqual(2,  par1);
      Assert.IsTrue(res2);
      Assert.AreEqual(-1, par2);
      Assert.IsFalse(res3);
      Assert.AreEqual(0, par3);
    }

    [TestMethod]
    public void ComputingNode_TrySetParam()
    {
      var node = new ScalarProductNode(2, -1);
      node.Build();

      var res1 = node.TrySetParam(0, 3, false);
      var res2 = node.TrySetParam(1, -2, true);
      var res3 = node.TrySetParam(2, 5, true);

      Assert.IsTrue(res1);
      Assert.IsTrue(res2);
      Assert.IsFalse(res3);

      Assert.AreEqual(3,  node.Coeffs[0]);
      Assert.AreEqual(-3, node.Coeffs[1]);
    }

    [TestMethod]
    public void ComputingNode_TryUpdateParams()
    {
      var node = new ScalarProductNode(2, -1);
      node.Build();
      var pars = new double[] { 1, 2, 3, -4, -1, 3 };
      var cursor = 1;

      var res = node.TryUpdateParams(pars, false, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(3, cursor);
      Assert.AreEqual(2, node.Coeffs[0]);
      Assert.AreEqual(3, node.Coeffs[1]);

      res = node.TryUpdateParams(pars, true, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(5, cursor);
      Assert.AreEqual(-2, node.Coeffs[0]);
      Assert.AreEqual(2, node.Coeffs[1]);

      res = node.TryUpdateParams(pars, false, ref cursor);
      Assert.IsFalse(res);
    }

    #endregion

    #region AggregateNode

    [TestMethod]
    [ExpectedException(typeof(MLException))]
    public void AggregateNode_Build_NoSubNodes()
    {
      var node = new AggregateNode<double, double>();
      node.SetMergeNode(new MergeNode(2, -1));

      node.Build();
    }

    [TestMethod]
    public void AggregateNode_Build()
    {
      var node = new AggregateNode<double, double>();
      var merge = new MergeNode(2, -1);
      var sub1 = new DoublingNode();
      var sub2 = new ShiftingNode(3);
      node.AddSubNode(sub1);
      node.AddSubNode(sub2);
      node.SetMergeNode(merge);

      node.Build();

      Assert.AreEqual(0, node.ParamCount);
      Assert.AreEqual(0, node.SubNodes[0].ParamCount);
      Assert.AreEqual(1, node.SubNodes[1].ParamCount);
      Assert.AreEqual(2, node.MergeNode.ParamCount);
    }

    [TestMethod]
    public void AggregateNode_Calculate()
    {
      var node = new AggregateNode<double, double>();
      var merge = new MergeNode(2, -1);
      var sub1 = new DoublingNode();
      var sub2 = new ShiftingNode(3);
      node.AddSubNode(sub1);
      node.AddSubNode(sub2);
      node.SetMergeNode(merge);

      node.Build();

      var res = node.Calculate(2);

      Assert.AreEqual(3, res);
    }

    [TestMethod]
    public void AggregateNode_TryGetParam()
    {
      var node = new AggregateNode<double, double>();
      var sub1 = new DoublingNode();
      var sub2 = new ShiftingNode(3);
      var merge = new MergeNode(2, -1);
      node.AddSubNode(sub1);
      node.AddSubNode(sub2);
      node.SetMergeNode(merge);

      node.Build();

      double par1;
      double par2;
      double par3;
      double par4;
      var res1 = node.TryGetParam(0, out par1);
      var res2 = node.TryGetParam(1, out par2);
      var res3 = node.TryGetParam(2, out par3);
      var res4 = node.TryGetParam(3, out par4);

      Assert.IsTrue(res1);
      Assert.AreEqual(3, par1);
      Assert.IsTrue(res2);
      Assert.AreEqual(2, par2);
      Assert.IsTrue(res3);
      Assert.AreEqual(-1, par3);
      Assert.IsFalse(res4);
      Assert.AreEqual(0, par4);
    }

    [TestMethod]
    public void AggregateNode_TrySetParam()
    {
      var node = new AggregateNode<double, double>();
      var sub1 = new DoublingNode();
      var sub2 = new ShiftingNode(3);
      var merge = new MergeNode(2, -1);
      node.AddSubNode(sub1);
      node.AddSubNode(sub2);
      node.SetMergeNode(merge);

      node.Build();

      var res1 = node.TrySetParam(0,  1, false);
      var res2 = node.TrySetParam(1,  1, true);
      var res3 = node.TrySetParam(2, -2, false);
      var res4 = node.TrySetParam(3, -3, true);

      Assert.IsTrue(res1);
      Assert.AreEqual(1, sub2.Shift);
      Assert.IsTrue(res2);
      Assert.AreEqual(3, merge.Coeffs[0]);
      Assert.IsTrue(res3);
      Assert.AreEqual(-2, merge.Coeffs[1]);
      Assert.IsFalse(res4);
    }

    [TestMethod]
    public void AggregateNode_TryUpdateParams()
    {
      var node = new AggregateNode<double, double>();
      var sub1 = new DoublingNode();
      var sub2 = new ShiftingNode(3);
      var merge = new MergeNode(2, -1);
      node.AddSubNode(sub1);
      node.AddSubNode(sub2);
      node.SetMergeNode(merge);

      node.Build();
      var pars = new double[] { 1, 2, 3, -4, -1, 3, 1, -1 };
      var cursor = 1;

      var res = node.TryUpdateParams(pars, false, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(4,  cursor);
      Assert.AreEqual(2,  sub2.Shift);
      Assert.AreEqual(3,  merge.Coeffs[0]);
      Assert.AreEqual(-4, merge.Coeffs[1]);

      res = node.TryUpdateParams(pars, true, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(7,  cursor);
      Assert.AreEqual(1,  sub2.Shift);
      Assert.AreEqual(6,  merge.Coeffs[0]);
      Assert.AreEqual(-3, merge.Coeffs[1]);

      res = node.TryUpdateParams(pars, true, ref cursor);
      Assert.IsFalse(res);
    }

    #endregion

    #region CompositeNode

    [TestMethod]
    public void CompositeNode_Build()
    {
      var node = new CompositeNode<double, double>();
      var sub1 = new DoublingNode();
      var sub2 = new ShiftingNode(3);
      var sub3 = new PolynomialNode(1, new double[] { 2, -3 });
      node.AddSubNode(sub1);
      node.AddSubNode(sub2);
      node.AddSubNode(sub3);

      node.Build();

      Assert.AreEqual(0, node.ParamCount);
      Assert.AreEqual(0, node.SubNodes[0].ParamCount);
      Assert.AreEqual(1, node.SubNodes[1].ParamCount);
      Assert.AreEqual(3, node.SubNodes[2].ParamCount);
    }

    [TestMethod]
    public void CompositeNode_Calculate()
    {
      var node = new CompositeNode<double, double>();
      var sub1 = new DoublingNode();
      var sub2 = new ShiftingNode(3);
      var sub3 = new PolynomialNode(1, new double[] { 2, -3 });
      node.AddSubNode(sub1);
      node.AddSubNode(sub2);
      node.AddSubNode(sub3);

      node.Build();

      var res = node.Calculate(2);

      Assert.AreEqual(3,  res.Length);
      Assert.AreEqual(4,  res[0]);
      Assert.AreEqual(5,  res[1]);
      Assert.AreEqual(-4, res[2]);
    }

    [TestMethod]
    public void CompositeNode_TryGetParam()
    {
      var node = new CompositeNode<double, double>();
      var sub1 = new DoublingNode();
      var sub2 = new ShiftingNode(3);
      var sub3 = new PolynomialNode(1, new double[] { 2, -3 });
      node.AddSubNode(sub1);
      node.AddSubNode(sub2);
      node.AddSubNode(sub3);

      node.Build();

      double par1;
      double par2;
      double par3;
      double par4;
      double par5;
      var res1 = node.TryGetParam(0, out par1);
      var res2 = node.TryGetParam(1, out par2);
      var res3 = node.TryGetParam(2, out par3);
      var res4 = node.TryGetParam(3, out par4);
      var res5 = node.TryGetParam(4, out par5);

      Assert.IsTrue(res1);
      Assert.AreEqual(3, par1);
      Assert.IsTrue(res2);
      Assert.AreEqual(1, par2);
      Assert.IsTrue(res3);
      Assert.AreEqual(2, par3);
      Assert.IsTrue(res4);
      Assert.AreEqual(-3, par4);
      Assert.IsFalse(res5);
      Assert.AreEqual(0, par5);
    }

    [TestMethod]
    public void CompositeNode_TrySetParam()
    {
      var node = new CompositeNode<double, double>();
      var sub1 = new DoublingNode();
      var sub2 = new ShiftingNode(3);
      var sub3 = new PolynomialNode(1, new double[] { 2, -3});
      node.AddSubNode(sub1);
      node.AddSubNode(sub2);
      node.AddSubNode(sub3);

      node.Build();

      var res1 = node.TrySetParam(0,  1, false);
      var res2 = node.TrySetParam(1,  1, true);
      var res3 = node.TrySetParam(2, -2, false);
      var res4 = node.TrySetParam(3, -3, true);
      var res5 = node.TrySetParam(4, 4,  false);

      Assert.IsTrue(res1);
      Assert.AreEqual(1, sub2.Shift);
      Assert.IsTrue(res2);
      Assert.AreEqual(2, sub3.Degree);
      Assert.IsTrue(res3);
      Assert.AreEqual(-2, sub3.Coeffs[0]);
      Assert.IsTrue(res4);
      Assert.AreEqual(-6, sub3.Coeffs[1]);
      Assert.IsFalse(res5);
    }

    [TestMethod]
    public void CompositeNode_TryUpdateParams()
    {
      var node = new CompositeNode<double, double>();
      var sub1 = new DoublingNode();
      var sub2 = new ShiftingNode(3);
      var sub3 = new PolynomialNode(1, new double[] { 2, -3 });
      node.AddSubNode(sub1);
      node.AddSubNode(sub2);
      node.AddSubNode(sub3);

      node.Build();
      var pars = new double[] { 1, 2, 3, -4, -1, 3, 1, -1, 2, 5 };
      var cursor = 1;

      var res = node.TryUpdateParams(pars, false, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(5,  cursor);
      Assert.AreEqual(2,  sub2.Shift);
      Assert.AreEqual(3,  sub3.Degree);
      Assert.AreEqual(-4, sub3.Coeffs[0]);
      Assert.AreEqual(-1, sub3.Coeffs[1]);

      res = node.TryUpdateParams(pars, true, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(9,  cursor);
      Assert.AreEqual(5,  sub2.Shift);
      Assert.AreEqual(4,  sub3.Degree);
      Assert.AreEqual(-5, sub3.Coeffs[0]);
      Assert.AreEqual(1,  sub3.Coeffs[1]);

      res = node.TryUpdateParams(pars, true, ref cursor);
      Assert.IsFalse(res);
    }

    #endregion

    #region JoinNode

    [TestMethod]
    [ExpectedException(typeof(MLException))]
    public void JoinNode_Build_NoInputNode()
    {
      var node = new JoinNode<double, double, double>();
      var output = new DoublingNode();
      node.SetOutputNode(output);

      node.Build();
    }

    [TestMethod]
    [ExpectedException(typeof(MLException))]
    public void JoinNode_Build_NoOutputNode()
    {
      var node = new JoinNode<double, double, double>();
      var input = new DoublingNode();
      node.SetInputNode(input);

      node.Build();
    }

    [TestMethod]
    public void JoinNode_Build()
    {
      var node1 = new DoublingNode();
      var node2 = new ShiftingNode(4);
      var node3 = new PowerNode(2);
      var node = new JoinNode<double, double, double>();
      node.SetInputNode(node1);
      node.SetOutputNode(node2);

      node.Build();

      Assert.AreEqual(0, node.ParamCount);
      Assert.AreEqual(0, node.InputNode.ParamCount);
      Assert.AreEqual(1, node.OutputNode.ParamCount);

      node = new JoinNode<double, double, double>();
      node.SetInputNode(node3);
      node.SetOutputNode(node1);

      node.Build();

      Assert.AreEqual(0, node.ParamCount);
      Assert.AreEqual(1, node.InputNode.ParamCount);
      Assert.AreEqual(0, node.OutputNode.ParamCount);

      node = new JoinNode<double, double, double>();
      node.SetInputNode(node2);
      node.SetOutputNode(node3);

      node.Build();

      Assert.AreEqual(0, node.ParamCount);
      Assert.AreEqual(1, node.InputNode.ParamCount);
      Assert.AreEqual(1, node.OutputNode.ParamCount);
    }

    [TestMethod]
    public void JoinNode_Calculate()
    {
      var node1 = new DoublingNode();
      var node2 = new ShiftingNode(4);
      var node = new JoinNode<double, double, double>();
      node.SetInputNode(node1);
      node.SetOutputNode(node2);

      node.Build();

      var res = node.Calculate(3);

      Assert.AreEqual(10, res);
    }

    [TestMethod]
    public void JoinNode_TryGetParam()
    {
      var node1 = new DoublingNode();
      var node2 = new ShiftingNode(4);
      var node = new JoinNode<double, double, double>();
      node.SetInputNode(node1);
      node.SetOutputNode(node2);

      node.Build();

      double par1;
      double par2;
      var res1 = node.TryGetParam(0, out par1);
      var res2 = node.TryGetParam(1, out par2);

      Assert.IsTrue(res1);
      Assert.AreEqual(4, par1);
      Assert.IsFalse(res2);
    }

    [TestMethod]
    public void JoinNode_TrySetParam()
    {
      var node1 = new PolynomialNode(2, new[] { 1.0D, -1.0D, 2.0D });
      var node2 = new ShiftingNode(4);
      var node = new JoinNode<double, double, double>();
      node.SetInputNode(node1);
      node.SetOutputNode(node2);

      node.Build();

      var res1 = node.TrySetParam(0,  1, false);
      var res2 = node.TrySetParam(1,  1, true);
      var res3 = node.TrySetParam(2, -2, false);
      var res4 = node.TrySetParam(3, -1, true);
      var res5 = node.TrySetParam(4,  3, false);
      var res6 = node.TrySetParam(5,  4, true);

      Assert.IsTrue(res1);
      Assert.AreEqual(1, node1.Degree);
      Assert.IsTrue(res2);
      Assert.AreEqual(2, node1.Coeffs[0]);
      Assert.IsTrue(res4);
      Assert.AreEqual(-2, node1.Coeffs[1]);
      Assert.IsTrue(res4);
      Assert.AreEqual(1, node1.Coeffs[2]);
      Assert.IsTrue(res5);
      Assert.AreEqual(3, node2.Shift);
      Assert.IsFalse(res6);
    }

    [TestMethod]
    public void JoinNode_TryUpdateParams()
    {
      var node1 = new PolynomialNode(2, new[] { 1.0D, -1.0D, 2.0D });
      var node2 = new ShiftingNode(4);
      var node = new JoinNode<double, double, double>();
      node.SetInputNode(node1);
      node.SetOutputNode(node2);

      node.Build();

      var pars = new double[] { 1, 2, 3, -4, -1, 3, 1, -1, 2, -2, 1, 2, 2, 4 };
      var cursor = 1;

      var res = node.TryUpdateParams(pars, false, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(6,  cursor);
      Assert.AreEqual(2,  node1.Degree);
      Assert.AreEqual(3,  node1.Coeffs[0]);
      Assert.AreEqual(-4, node1.Coeffs[1]);
      Assert.AreEqual(-1, node1.Coeffs[2]);
      Assert.AreEqual(3,  node2.Shift);

      res = node.TryUpdateParams(pars, true, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(11, cursor);
      Assert.AreEqual(3,  node1.Degree);
      Assert.AreEqual(2,  node1.Coeffs[0]);
      Assert.AreEqual(-2, node1.Coeffs[1]);
      Assert.AreEqual(-3, node1.Coeffs[2]);
      Assert.AreEqual(4,  node2.Shift);

      res = node.TryUpdateParams(pars, false, ref cursor);
      Assert.IsFalse(res);
    }

    #endregion

    #region SequenceNode

    [TestMethod]
    public void SequenceNode_Build()
    {
      var node1 = new DoublingNode();
      var node2 = new ShiftingNode(4);
      var node3 = new PowerNode(2);
      var node = new SequenceNode<double>();
      node.AddSubNode(node1);
      node.AddSubNode(node2);
      node.AddSubNode(node3);

      node.Build();

      Assert.AreEqual(0, node.ParamCount);
      Assert.AreEqual(3, node.SubNodes.Length);
      Assert.AreEqual(0, node.SubNodes[0].ParamCount);
      Assert.AreEqual(1, node.SubNodes[1].ParamCount);
      Assert.AreEqual(1, node.SubNodes[2].ParamCount);
    }

    [TestMethod]
    public void SequenceNode_Calculate()
    {
      var node1 = new DoublingNode();
      var node2 = new ShiftingNode(4);
      var node3 = new PowerNode(2);
      var node = new SequenceNode<double>();
      node.AddSubNode(node1);
      node.AddSubNode(node2);
      node.AddSubNode(node3);

      node.Build();

      var res = node.Calculate(3);

      Assert.AreEqual(100, res);
    }

    [TestMethod]
    public void SequenceNode_TryGetParam()
    {
      var node1 = new DoublingNode();
      var node2 = new ShiftingNode(4);
      var node3 = new PowerNode(2);
      var node = new SequenceNode<double>();
      node.AddSubNode(node1);
      node.AddSubNode(node2);
      node.AddSubNode(node3);

      node.Build();

      double par1;
      double par2;
      double par3;
      var res1 = node.TryGetParam(0, out par1);
      var res2 = node.TryGetParam(1, out par2);
      var res3 = node.TryGetParam(2, out par3);

      Assert.IsTrue(res1);
      Assert.AreEqual(4, par1);
      Assert.IsTrue(res2);
      Assert.AreEqual(2, par2);
      Assert.IsFalse(res3);
    }

    [TestMethod]
    public void SequenceNode_TrySetParam()
    {
      var node1 = new PolynomialNode(2, new[] { 1.0D, -1.0D, 2.0D });
      var node2 = new ShiftingNode(4);
      var node3 = new PowerNode(2);
      var node = new SequenceNode<double>();
      node.AddSubNode(node1);
      node.AddSubNode(node2);
      node.AddSubNode(node3);

      node.Build();

      var res1 = node.TrySetParam(0,  1, false);
      var res2 = node.TrySetParam(1,  1, true);
      var res3 = node.TrySetParam(2, -2, false);
      var res4 = node.TrySetParam(3, -1, true);
      var res5 = node.TrySetParam(4,  3, false);
      var res6 = node.TrySetParam(5,  4, true);
      var res7 = node.TrySetParam(6,  -4, true);

      Assert.IsTrue(res1);
      Assert.AreEqual(1, node1.Degree);
      Assert.IsTrue(res2);
      Assert.AreEqual(2, node1.Coeffs[0]);
      Assert.IsTrue(res4);
      Assert.AreEqual(-2, node1.Coeffs[1]);
      Assert.IsTrue(res4);
      Assert.AreEqual(1, node1.Coeffs[2]);
      Assert.IsTrue(res5);
      Assert.AreEqual(3, node2.Shift);
      Assert.IsTrue(res6);
      Assert.AreEqual(6, node3.Power);
      Assert.IsFalse(res7);
    }

    [TestMethod]
    public void SequenceNode_TryUpdateParams()
    {
      var node1 = new PolynomialNode(2, new[] { 1.0D, -1.0D, 2.0D });
      var node2 = new ShiftingNode(4);
      var node3 = new PowerNode(2);
      var node = new SequenceNode<double>();
      node.AddSubNode(node1);
      node.AddSubNode(node2);
      node.AddSubNode(node3);

      node.Build();

      var pars = new double[] { 1, 2, 3, -4, -1, 3, 1, -1, 2, -2, 1, 2, 2, 4 };
      var cursor = 1;

      var res = node.TryUpdateParams(pars, false, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(7,  cursor);
      Assert.AreEqual(2,  node1.Degree);
      Assert.AreEqual(3,  node1.Coeffs[0]);
      Assert.AreEqual(-4, node1.Coeffs[1]);
      Assert.AreEqual(-1, node1.Coeffs[2]);
      Assert.AreEqual(3,  node2.Shift);
      Assert.AreEqual(1,  node3.Power);

      res = node.TryUpdateParams(pars, true, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(13, cursor);
      Assert.AreEqual(1,  node1.Degree);
      Assert.AreEqual(5,  node1.Coeffs[0]);
      Assert.AreEqual(-6, node1.Coeffs[1]);
      Assert.AreEqual(0, node1.Coeffs[2]);
      Assert.AreEqual(5,  node2.Shift);
      Assert.AreEqual(3,  node3.Power);

      res = node.TryUpdateParams(pars, false, ref cursor);
      Assert.IsFalse(res);
    }

    #endregion

    #region ComputingNetwork

    [TestMethod]
    public void ComputingNetwork_Build()
    {
      var net = new SmallNetwork();
      net.Build();

      Assert.AreEqual(0, net.ParamCount);
      Assert.AreEqual(0, ((dynamic)net).InputNode.ParamCount);
      Assert.AreEqual(0, ((dynamic)net).OutputNode.ParamCount);
      Assert.AreEqual(0, ((dynamic)net).OutputNode.SubNodes[0].SubNodes[0].ParamCount);
      Assert.AreEqual(0, ((dynamic)net).OutputNode.SubNodes[0].SubNodes[0].SubNodes[0].ParamCount);
      Assert.AreEqual(1, ((dynamic)net).OutputNode.SubNodes[0].SubNodes[0].SubNodes[1].ParamCount);
      Assert.AreEqual(1, ((dynamic)net).OutputNode.SubNodes[0].SubNodes[1].ParamCount);
      Assert.AreEqual(2, ((dynamic)net).OutputNode.SubNodes[0].MergeNode.ParamCount);
      Assert.AreEqual(1, ((dynamic)net).OutputNode.SubNodes[1].ParamCount);
    }

    [TestMethod]
    public void ComputingNetwork_Calculate()
    {
      var net = new SmallNetwork();
      net.Build();

      var input1 = new Point2D(3, 4);
      var input2 = new Point2D(3, 2);
      var res1 = net.Calculate(input1);
      var res2 = net.Calculate(input2);

      Assert.AreEqual(-7, res1);
      Assert.AreEqual(-4, res2);
    }

    [TestMethod]
    public void ComputingNetwork_TryGetParam()
    {
      var net = new SmallNetwork();
      net.Build();

      double w1;
      double w2;
      double w3;
      double w4;
      double w5;
      double w6;
      var res1 = net.TryGetParam(0,  out w1);
      var res2 = net.TryGetParam(1,  out w2);
      var res3 = net.TryGetParam(2,  out w3);
      var res4 = net.TryGetParam(3,  out w4);
      var res5 = net.TryGetParam(4,  out w5);
      var res6 = net.TryGetParam(5,  out w6);

      Assert.IsTrue(res1);
      Assert.AreEqual(-1, w1);
      Assert.IsTrue(res2);
      Assert.AreEqual(2, w2);
      Assert.IsTrue(res3);
      Assert.AreEqual(2, w3);
      Assert.IsTrue(res4);
      Assert.AreEqual(-1, w4);
      Assert.IsTrue(res5);
      Assert.AreEqual(3, w5);
      Assert.IsFalse(res6);
      Assert.AreEqual(0, w6);
    }

    [TestMethod]
    public void ComputingNetwork_TrySetParam()
    {
      var net = new SmallNetwork();
      net.Build();

      var res1 = net.TrySetParam(0, 1, false);
      var res2 = net.TrySetParam(1, 2, true);
      var res3 = net.TrySetParam(2, 3, false);
      var res4 = net.TrySetParam(3, 4, true);
      var res5 = net.TrySetParam(4, 5, false);
      var res6 = net.TrySetParam(5, 6, true);

      Assert.IsTrue(res1);
      Assert.AreEqual(1, ((dynamic)net).OutputNode.SubNodes[0].SubNodes[0].SubNodes[1].Shift);
      Assert.IsTrue(res2);
      Assert.AreEqual(4, ((dynamic)net).OutputNode.SubNodes[0].SubNodes[1].Power);
      Assert.IsTrue(res3);
      Assert.AreEqual(3, ((dynamic)net).OutputNode.SubNodes[0].MergeNode.Coeffs[0]);
      Assert.IsTrue(res4);
      Assert.AreEqual(3, ((dynamic)net).OutputNode.SubNodes[0].MergeNode.Coeffs[1]);
      Assert.IsTrue(res5);
      Assert.AreEqual(5, ((dynamic)net).OutputNode.SubNodes[1].Shift);
      Assert.IsFalse(res6);
    }

    [TestMethod]
    public void ComputingNetwork_TryUpdateParams()
    {
      var net = new SmallNetwork();
      net.Build();
      var pars = new double[] { 1, 2, 3, -4, -1, 3, 1, -1, 2, -2, 1, 2 };
      var cursor = 1;

      var res = net.TryUpdateParams(pars, false, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(6,  cursor);
      Assert.AreEqual(2,  ((dynamic)net).OutputNode.SubNodes[0].SubNodes[0].SubNodes[1].Shift);
      Assert.AreEqual(3,  ((dynamic)net).OutputNode.SubNodes[0].SubNodes[1].Power);
      Assert.AreEqual(-4, ((dynamic)net).OutputNode.SubNodes[0].MergeNode.Coeffs[0]);
      Assert.AreEqual(-1, ((dynamic)net).OutputNode.SubNodes[0].MergeNode.Coeffs[1]);
      Assert.AreEqual(3,  ((dynamic)net).OutputNode.SubNodes[1].Shift);

      res = net.TryUpdateParams(pars, true, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(11, cursor);
      Assert.AreEqual(3,  ((dynamic)net).OutputNode.SubNodes[0].SubNodes[0].SubNodes[1].Shift);
      Assert.AreEqual(2,  ((dynamic)net).OutputNode.SubNodes[0].SubNodes[1].Power);
      Assert.AreEqual(-2, ((dynamic)net).OutputNode.SubNodes[0].MergeNode.Coeffs[0]);
      Assert.AreEqual(-3, ((dynamic)net).OutputNode.SubNodes[0].MergeNode.Coeffs[1]);
      Assert.AreEqual(4,  ((dynamic)net).OutputNode.SubNodes[1].Shift);

      res = net.TryUpdateParams(pars, false, ref cursor);
      Assert.IsFalse(res);
    }

    #endregion

    #region Large ComputingNetwork

    [TestMethod]
    public void Large_ComputingNetwork_Build()
    {
      var dim = 100;
      var lcount = 10;
      var net = new LargeNetwork(dim, lcount);
      net.Build();

      Assert.AreEqual(0, net.ParamCount);

      for (int i=0; i<lcount-1; i++)
      {
        var node = ((dynamic)net).InputNode.SubNodes[i];
        Assert.AreEqual(10000, node.ParamCount);
      }
      Assert.AreEqual(100, net.OutputNode.ParamCount);
    }

    [TestMethod]
    public void Large_ComputingNetwork_TryGetSetParam()
    {
      var dim = 100;
      var lcount = 10;
      var pcount = dim*dim*lcount+dim;
      var net = new LargeNetwork(dim, lcount);
      net.Build();

      // raw

      for (int i=0; i<pcount+10; i++)
      {
        var result = net.TrySetParam(i, i, false);
        if (i<pcount) Assert.IsTrue(result);
        else Assert.IsFalse(result);
      }

      for (int i=0; i<pcount+10; i++)
      {
        double value;
        var result = net.TryGetParam(i, out value);
        if (i<pcount)
        {
          Assert.IsTrue(result);
          Assert.AreEqual(value, i);
        }
        else
          Assert.IsFalse(result);
      }

      // delta

      for (int i=0; i<pcount+10; i++)
      {
        var result = net.TrySetParam(i, i, true);
        if (i<pcount) Assert.IsTrue(result);
        else Assert.IsFalse(result);
      }

      for (int i=0; i<pcount+10; i++)
      {
        double value;
        var result = net.TryGetParam(i, out value);
        if (i<pcount)
        {
          Assert.IsTrue(result);
          Assert.AreEqual(value, 2*i);
        }
        else
          Assert.IsFalse(result);
      }
    }

    [TestMethod]
    public void Large_ComputingNetwork_TryUpdateParams()
    {
      var dim = 100;
      var lcount = 10;
      var pcount = dim*dim*lcount+dim;
      var net = new LargeNetwork(dim, lcount);
      net.Build();

      var pars = new double[pcount];
      for (int i=0; i<pcount; i++)
        pars[i] = i;

      // raw

      int cursor = 0;
      var result = net.TryUpdateParams(pars, false, ref cursor);
      Assert.IsTrue(result);
      Assert.AreEqual(pcount, cursor);

      for (int i=0; i<pcount; i++)
      {
        double value;
        result = net.TryGetParam(i, out value);
        Assert.IsTrue(result);
        Assert.AreEqual(value, i);
      }

      // delta

      cursor = 0;
      result = net.TryUpdateParams(pars, true, ref cursor);
      Assert.IsTrue(result);
      Assert.AreEqual(pcount, cursor);

      for (int i=0; i<pcount; i++)
      {
        double value;
        result = net.TryGetParam(i, out value);
        Assert.IsTrue(result);
        Assert.AreEqual(value, 2*i);
      }
    }

    [TestMethod]
    public void Large_Bench_ComputingNetwork_Calculate()
    {
      var dim = 512;   // dim x dim matrices
      var lcount = 20; // lcount layers
      var net = new LargeNetwork(dim, lcount);
      net.Build();

      var input = new double[dim];
      for (int i=0; i<dim; i++)
        input[i] = 0.0001D;

      var timer = new Stopwatch();
      var times = 1000;
      timer.Start();

      for (int i=0; i<times; i++)
        net.Calculate(input);

      timer.Stop();
      Console.WriteLine("Calculation BM: (dim={0} nodes={1}): {2}ms",
                        dim,
                        lcount,
                        (int)((float)timer.ElapsedMilliseconds/times));
    }

    [TestMethod]
    public void Large_Bench_ComputingNetwork_BulkSetVSIndexSet()
    {
      var dim = 512;   // dim x dim matrices
      var lcount = 10; // lcount layers
      var pcount = dim*dim*lcount+dim;
      var net = new LargeNetwork(dim, lcount);
      net.Build();

      var pars = new double[pcount];
      pars[12345] = 1.5D;
      pars[2345] = 0.5D;
      pars[73450] = 3.5D;

      var timer = new Stopwatch();
      var times = 100;
      timer.Start();

      for (int i=0; i<times; i++)
      {
        int cursor = 0;
        net.TryUpdateParams(pars, false, ref cursor);
      }

      timer.Stop();
      Console.WriteLine("Bulk Set BM: (dim={0} nodes={1}): {2} ticks",
                        dim,
                        lcount,
                        (int)((float)timer.ElapsedTicks/times));

      timer.Reset();
      timer.Start();
      for (int i=0; i<times; i++)
      {
        net.TrySetParam(12345, 1.5D, false);
        net.TrySetParam(2345, 0.5D, false);
        net.TrySetParam(73450, 3.5D, false);
      }

      timer.Stop();
      Console.WriteLine("Index Set BM: (dim={0} nodes={1}): {2} ticks",
                        dim,
                        lcount,
                        (int)((float)timer.ElapsedTicks/times));
    }

    #endregion
  }
}
