using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NFX;
using NFX.Serialization.JSON;

namespace ML.Core.ComputingNetworks
{
  public abstract class _ComputingNode<TVal>: INamed
  {
    private readonly string m_Name;
    private readonly bool m_IsInput;
    private readonly bool m_IsOutput;
    private readonly _ComputingNode<TVal>[] m_Inputs;
    private readonly _ComputingNode<TVal>[] m_Outputs;
    private double[] m_Parameters;

    private int m_Index;
    private bool m_Built;

    public _ComputingNode(IEnumerable<_ComputingNode<TVal>> inputs, IEnumerable<_ComputingNode<TVal>> outputs, string name)
    {
      if (inputs  != null) m_Inputs= inputs.ToArray();
      else m_IsInput = true;

      if (outputs != null) m_Outputs = outputs.ToArray();
      else m_IsOutput = true;

      m_Index = -1;
    }


    public string Name { get { return m_Name; } }
    /// <summary>
    /// Returns number of node parameters
    /// </summary>
    public abstract int ParamCount { get; }

    public double[] Parameters { get { return m_Parameters; } }
    public _ComputingNode<TVal>[] Inputs  { get { return m_Inputs; } }
    public _ComputingNode<TVal>[] Outputs { get { return m_Outputs; } }
    public int  Index       { get { return m_Index; } }
    public bool IsInput     { get { return m_IsInput;  } }
    public bool IsOutput    { get { return m_IsOutput; } }
    public bool IsTrainable { get; set; }


    public void Build()
    {
      if (m_Built) return;

      m_Parameters = new double[ParamCount];

      DoBuild();
      if (!m_IsOutput)
      {
        foreach (var on in m_Outputs)
          on.Build();
      }

      m_Built = true;
    }

    public void Calculate(TVal[] inputValues, TVal result)
    {
      if ((IsInput && inputValues.Length != 1) ||
          (!IsInput && inputValues.Length != Inputs.Length))
        throw new MLException("Incorrect lengths");

      DoCalculate(result);

      if (!IsOutput)
      {
        var outputValues = new TVal[Outputs.Length];

      }
    }

    public virtual void Serialize(Stream stream)
    {
      if (!m_Built) throw new MLException("Node is not built");

      var body = new JSONDataMap();
      body.Add("type", this.GetType().AssemblyQualifiedName);
      body.Add("name", Name);
      body.Add("index", Index);
      body.Add("is-input", IsInput);
      body.Add("is-output", IsOutput);
      body.Add("is-trainable", IsTrainable);
      body.Add("param-count", ParamCount);

      var inputs  = m_Inputs  ?? new _ComputingNode<TVal>[0];
      var outputs = m_Outputs ?? new _ComputingNode<TVal>[0];
      body.Add("inputs", new JSONDataArray(inputs));
      body.Add("outputs", new JSONDataArray(outputs));

      DoSerialize(body);

      // TODO: store parameters separately in HD5 or CDF?
      var paramBytes = new byte[ParamCount * sizeof(double)];
      Buffer.BlockCopy(Parameters, 0, paramBytes, 0, paramBytes.Length);
      var parameters = Convert.ToBase64String(paramBytes);
      body.Add("parameters", parameters);
      // TODO: store parameters separately in HD5 or CDF?

      var root = new JSONDataMap { { "node", body } };
      JSONWriter.Write(root, stream);
    }

    protected abstract void DoBuild();

    protected abstract void DoSerialize(JSONDataMap body);

    protected abstract void DoCalculate(TVal result);
  }
}
