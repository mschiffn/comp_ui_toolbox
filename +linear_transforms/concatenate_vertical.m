%
% concatenate linear transforms vertically
% author: Martin Schiffner
% date: 2016-08-10
% modified: 2018-04-21
%
classdef concatenate_vertical < linear_transforms.linear_transform

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)
        N_transforms
        size_transforms
        transforms
        indices_coef_start
        indices_coef_stop
        str_names_single
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function LT_concatenate_vertical = concatenate_vertical( varargin )

            % number of concatenateed linear transforms
            N_transforms_temp = nargin;

            % check individual transforms for validity and size
            size = zeros( 2, N_transforms_temp );
            for index_transform = 1:N_transforms_temp

                % check class of input argument
                if ~isa( varargin{ index_transform }, 'linear_transforms.linear_transform' )
                    errorStruct.message     = sprintf( 'Input %d is not an instance of linear_transforms.linear_transform!', index_transform );
                    errorStruct.identifier	= 'LT_concatenate_vertical:TypeMismatch';
                    error( errorStruct );
                end
                % assertion: varargin{ index_transform } is an instance of a linear transform

                % get size of selected transform
                size( :, index_transform ) = varargin{ index_transform }.size_transform;

                % create name string
                if index_transform == 1
                    str_name = sprintf( '%s', varargin{ index_transform }.str_name );
                else
                    % check correct size of linear transform for concatenating
                    if size( 2, index_transform ) ~= size( 2, index_transform - 1 )
                        fprintf('incorrect size of linear transform\n');
                        return
                    end
                    str_name = sprintf( '%s_%s', str_name, varargin{ index_transform }.str_name );
                end
            end % for index_transform = 1:N_transforms_temp

            % size of vertically concatenated forward transform
            N_coefficients	= sum( size( 1, : ), 2 );
            N_lattice       = size( 2, 1 );

            % constructor of superclass
            LT_concatenate_vertical@linear_transforms.linear_transform( N_coefficients, N_lattice, str_name );

            % internal properties
            LT_concatenate_vertical.N_transforms        = N_transforms_temp;
            LT_concatenate_vertical.transforms          = varargin;
            LT_concatenate_vertical.size_transforms     = size;
            LT_concatenate_vertical.indices_coef_stop	= cumsum( size( 1, :) );
            LT_concatenate_vertical.indices_coef_start	= [1, LT_concatenate_vertical.indices_coef_stop(1:end-1) + 1];

            LT_concatenate_vertical.str_names_single	= cell( 1, N_transforms_temp );
            for index_transform = 1:N_transforms_temp
                LT_concatenate_vertical.str_names_single{ index_transform } = varargin{ index_transform }.str_name;
            end

        end

        %------------------------------------------------------------------
        % overload method: forward transform
        %------------------------------------------------------------------
        function y = forward_transform( LT_concatenate_vertical, x )

            % initialize internal variables
            y_temp	= cell( 1, LT_concatenate_vertical.N_transforms );
            y       = zeros( LT_concatenate_vertical.N_coefficients, 1 );

            % apply vertically concatenated forward transforms
            for index_transform = 1:LT_concatenate_vertical.N_transforms

                % indices in output column vector
                index_start	= LT_concatenate_vertical.indices_coef_start( index_transform );
                index_stop	= LT_concatenate_vertical.indices_coef_stop( index_transform );

                y_temp{ index_transform }	= LT_concatenate_vertical.transforms{index_transform}.forward_transform( x );
                y( index_start:index_stop ) = y_temp{ index_transform }(:);
            end
        end

        %------------------------------------------------------------------
        % overload method: adjoint transform
        %------------------------------------------------------------------
        function y = adjoint_transform( LT_concatenate_vertical, x )

            y_temp	= cell( 1, LT_concatenate_vertical.N_transforms );
            y       = zeros( LT_concatenate_vertical.N_lattice, 1 );

            % apply horizontally concatenated adjoint transforms
            for index_transform = 1:LT_concatenate_vertical.N_transforms

                % indices in input column vector
                index_start	= LT_concatenate_vertical.indices_coef_start( index_transform );
                index_stop	= LT_concatenate_vertical.indices_coef_stop( index_transform );

                y_temp{ index_transform } = LT_concatenate_vertical.transforms{index_transform}.adjoint_transform( x(index_start:index_stop) );
                y = y + y_temp{ index_transform }(:);
            end
        end

    end % methods
    
end % classdef concatenate_vertical < linear_transforms.linear_transform