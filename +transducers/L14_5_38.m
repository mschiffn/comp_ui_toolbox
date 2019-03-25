%
% class for commercially available linear transducer array
% vendor name: Ultrasonix Medical Corporation
% model name: L14-5/38
%
% author: Martin F. Schiffner
% date: 2017-05-03
% modified: 2019-03-25
%
classdef L14_5_38 < transducers.array_planar

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = L14_5_38( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty N_dimensions
            if nargin >= 1 && ~isempty( varargin{ 1 } )
                N_dimensions = varargin{ 1 };
            else
                % specify two-dimensional transducer array as default
                N_dimensions = 2;
            end

            % ensure correct number of dimensions
            if ~( ( N_dimensions == 1 ) || ( N_dimensions == 2 ) )
                errorStruct.message     = 'Number of dimensions must equal either 1 or 2!';
                errorStruct.identifier	= 'L14_5_38:InvalidNumberOfDimensions';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            object@transducers.array_planar( transducers.parameters_L14_5_38, N_dimensions );

        end % function object = L14_5_38( varargin )

	end % methods

end % classdef L14_5_38 < transducers.array_planar
