using System;
using System.Collections.Generic;

namespace ML.Core.ComputingNetworks
{
  /// <summary>
  /// Represents computing node joined from two others -
  /// a node that sequentially tunells input through two layers
  ///
  /// TIn -> [[TIn -> THidOut] -> [THidOut -> THidOut] -> ... -> [THidOut -> TOut]] -> TOut
  ///            (input)               (hiddens)                      (output)
  /// </summary>
  /// <typeparam name="TIn">Input object type</typeparam>
  /// <typeparam name="THidOut">Hidden (inner) input object type</typeparam>
  /// <typeparam name="TOut">Output object type</typeparam>
  public sealed class SequenceNode<TIn, THidOut, TOut> : ComputingNode<TIn, TOut>
  {
    private ParamMultiIdx m_ParIdx;
    private ComputingNode<TIn, THidOut> m_InputNode;
    private ComputingNode<THidOut, TOut> m_OutputNode;
    private ComputingNode<THidOut, THidOut>[] m_HiddenNodes;


    public SequenceNode()
    {
      m_HiddenNodes = new ComputingNode<THidOut, THidOut>[0];
    }

    public override int ParamCount { get { return 0; } }
    public IComputingNode<TIn, THidOut> InputNode { get { return m_InputNode; } }
    public IComputingNode<THidOut, TOut> OutputNode { get { return m_OutputNode; } }
    public IComputingNode<THidOut, THidOut>[] HiddenNodes { get { return m_HiddenNodes; } }

    public void SetInputNode(ComputingNode<TIn, THidOut> inputNode)
    {
      if (inputNode==null)
        throw new MLException("Node can not be null");

      m_InputNode = inputNode;
    }

    public void SetOutputNode(ComputingNode<THidOut, TOut> outputNode)
    {
      if (outputNode==null)
        throw new MLException("Node can not be null");

      m_OutputNode = outputNode;
    }

    public void AddHidden(ComputingNode<THidOut, THidOut> hiddenNode)
    {
      if (hiddenNode==null)
        throw new MLException("Node can not be null");

      var len = m_HiddenNodes.Length;
      var hiddenNodes = new ComputingNode<THidOut, THidOut>[len+1];
      for (int i=0; i<len; i++)
        hiddenNodes[i] = m_HiddenNodes[i];
      hiddenNodes[len] = hiddenNode;

      m_HiddenNodes = hiddenNodes;
    }

    public override TOut Calculate(TIn input)
    {
      var innerResult = m_InputNode.Calculate(input);

      var len = m_HiddenNodes.Length;
      for (int i=0; i<len; i++)
        innerResult = m_HiddenNodes[i].Calculate(innerResult);

      return m_OutputNode.Calculate(innerResult);
    }

    /// <summary>
    /// Builds node before use (build search index etc.)
    /// </summary>
    internal override void DoBuild(bool buildIndex)
    {
      if (m_InputNode==null)
        throw new MLException("Input node has not been set");
      if (m_OutputNode==null)
        throw new MLException("Output node has not been set");

      if (buildIndex) BuildIndex(0);
      m_InputNode.DoBuild(false);

      var len = m_HiddenNodes.Length;
      for (int i=0; i<len; i++)
        m_HiddenNodes[i].DoBuild(false);

      m_OutputNode.DoBuild(false);
    }

    /// <summary>
    /// Builds fast search index
    /// </summary>
    /// <param name="startIdx">Index start value</param>
    /// <returns>Index end value</returns>
    internal override int BuildIndex(int startIdx)
    {
      if (m_InputNode==null)
        throw new MLException("Input node has not been set");
      if (m_OutputNode==null)
        throw new MLException("Output node has not been set");

      var len = m_HiddenNodes.Length;
      var idxs = new int[len+3];
      idxs[0] = startIdx;

      var endIdx = m_InputNode.BuildIndex(startIdx);
      idxs[1] = endIdx;

      for (int i=0; i<len; i++)
      {
        endIdx = m_HiddenNodes[i].BuildIndex(endIdx);
        idxs[i+2]=endIdx;
      }

      endIdx = m_OutputNode.BuildIndex(endIdx);
      idxs[len+2] = endIdx;

      m_ParIdx = new ParamMultiIdx(idxs);

      return endIdx;
    }

    /// <summary>
    /// Tries to return parameter value at some position
    /// WARNING: override this method carefully!
    /// Do not use base. as base methods operate with base class index which may differ from exact class index
    /// and so it will return wrong results. Override this method completely or not override
    /// </summary>
    /// <param name="idx">Linear index of the parameter</param>
    /// <param name="value">Parameter value</param>
    /// <returns>True is operation succeeded, false otherwise (unexisted index etc.)</returns>
    public override bool TryGetParam(int idx, out double value)
    {
      if (!m_ParIdx.CheckEnd(idx))
      {
        value = 0;
        return false;
      }

      if (m_ParIdx.CheckIdx(idx, 1))
        return m_InputNode.TryGetParam(idx, out value);

      var len = m_HiddenNodes.Length;
      for (int i=0; i<len; i++)
      {
        if (m_ParIdx.CheckIdx(idx, i+2))
          return m_HiddenNodes[i].TryGetParam(idx, out value);
      }

      return m_OutputNode.TryGetParam(idx, out value);
    }

    /// <summary>
    /// Tries to set parameter value at some position
    /// WARNING: override this method carefully!
    /// Do not use base. as base methods operate with base class index which may differ from exact class index
    /// and so it will return wrong results. Override this method completely or not override
    /// </summary>
    /// <param name="idx">Linear index of the parameter</param>
    /// <param name="value">Parameter value</param>
    /// <param name="isDelta">Is the values are exact or just delta to existing one</param>
    /// <returns>True is operation succeeded, false otherwise (unexisted index etc.)</returns>
    public override bool TrySetParam(int idx, double value, bool isDelta)
    {
      if (!m_ParIdx.CheckEnd(idx))
        return false;

      if (m_ParIdx.CheckIdx(idx, 1))
        return m_InputNode.TrySetParam(idx, value, isDelta);

      var len = m_HiddenNodes.Length;
      for (int i=0; i<len; i++)
      {
        if (m_ParIdx.CheckIdx(idx, i+2))
          return m_HiddenNodes[i].TrySetParam(idx, value, isDelta);
      }

      return m_OutputNode.TrySetParam(idx, value, isDelta);
    }

    /// <summary>
    /// Tries to update parameters of the network and passes it down to all sublayers
    /// WARNING: override this method carefully!
    /// Do not use base. as base methods operate with base class index which may differ from exact class index
    /// and so it will return wrong results. Override this method COMPLETELY or not override at all
    /// </summary>
    /// <param name="pars">Parameter values to update</param>
    /// <param name="isDelta">Is the values are exact or just deltas to existing ones</param>
    /// <param name="cursor">Start position in parameter vector</param>
    /// <returns>True is operation succeeded, false otherwise (bad parameter vector unexisted indices etc.)</returns>
    public override bool TryUpdateParams(double[] pars, bool isDelta, ref int cursor)
    {
      if (pars == null || pars.Length < cursor)
        return false;

      var success = m_InputNode.TryUpdateParams(pars, isDelta, ref cursor);
      if (!success) return false;

      var len = m_HiddenNodes.Length;
      for (int i=0; i<len; i++)
      {
        success = m_HiddenNodes[i].TryUpdateParams(pars, isDelta, ref cursor);
        if (!success) return false;
      }

      return m_OutputNode.TryUpdateParams(pars, isDelta, ref cursor);
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
