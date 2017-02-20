using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML.Core.ComputingNetworks
{
  public class ComputingNetwork<TIn, TOut> : ComputingNode<TIn, TOut>
  {
    private IComputingNode<TIn, TOut> m_Root;

    public ComputingNetwork()
    {
    }

    public IComputingNode<TIn, TOut> Root { get { return m_Root; } }

    public override int ParamCount { get { return 0; } }

    public void SetRoot(IComputingNode<TIn, TOut> root)
    {
      if (root==null)
        throw new MLException("Root can not be null");

      m_Root = root;
    }

    public override TOut Calculate(TIn input)
    {
      return m_Root.Calculate(input);
    }

    public override void Build()
    {
      m_Root.Build();
    }

    public override bool TryGetParam(int idx, out double value)
    {
      return m_Root.TryGetParam(idx, out value);
    }

    public override bool TrySetParam(int idx, double value, bool isDelta)
    {
      return m_Root.TrySetParam(idx, value, isDelta);
    }

    public override bool TryUpdateParams(double[] pars, bool isDelta, ref int cursor)
    {
      return m_Root.TryUpdateParams(pars, isDelta, ref cursor);
    }

    protected override double DoGetParam(int idx)
    {
      return 0;
    }

    protected override void DoSetParam(int idx, double value, bool isDelta)
    {
    }

    protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor)
    {
    }
  }
}
