﻿using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using NeuralNetwork;

namespace Labs.LabsFolder.Lab2
{
    public partial class FormCreate : Form
    {
        public int[] layerSizes;
        public TransferFunction[] TFunc;
        public FormCreate()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            layerSizes = new int[] { Convert.ToInt32(textBox1.Text), Convert.ToInt32(textBox2.Text), Convert.ToInt32(textBox3.Text) };
            TFunc = new TransferFunction[layerSizes.Length];

            for (int i = 0; i < TFunc.Length; i++)
            {
                TFunc[i] = (i == 0 ? TransferFunction.None : (TransferFunction)Enum.Parse(typeof(TransferFunction), comboBox1.Text));
            }
            this.Hide();
        }
    }
}
