%
% class for commercially available linear transducer array
% vendor name: Ultrasonix Medical Corporation
% model name: L14-5/38
%
% author: Martin F. Schiffner
% date: 2017-05-03
% modified: 2019-02-18
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
            % specify two-dimensional transducer array as default
            N_dimensions = 2;

            % process optional argument
            if numel( varargin ) > 0

                % ensure correct number of dimensions
                if ~( ( varargin{ 1 } == 1 ) || ( varargin{ 1 } == 2 ) )
                    errorStruct.message     = 'Number of dimensions N_dimensions must equal either 1 or 2!';
                    errorStruct.identifier	= 'L14_5_38:DimensionMismatch';
                    error( errorStruct );
                end
                N_dimensions = varargin{ 1 };
            end
            % assertion: N_dimensions == 1 || N_dimensions == 2

            %--------------------------------------------------------------
            % 2.) parameters of planar transducer array
            %--------------------------------------------------------------
            parameters = transducers.parameters_L14_5_38;

            %--------------------------------------------------------------
            % 3.) constructor of superclass
            %--------------------------------------------------------------
            object@transducers.array_planar( N_dimensions, parameters );

        end % function object = L14_5_38( varargin )

	end % methods

end % classdef L14_5_38 < transducers.array_planar
