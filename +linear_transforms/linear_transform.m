%
% superclass for all linear transforms
%
% author: Martin F. Schiffner
% date: 2016-08-12
% modified: 2019-05-20
%
classdef linear_transform

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        N_coefficients
        N_points

    end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = linear_transform( N_coefficients, N_points )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure positive integers for N_coefficients
            mustBeNonempty( N_coefficients );
            mustBePositive( N_coefficients );
            mustBeInteger( N_coefficients );

            % ensure positive integers for N_points
            mustBeNonempty( N_points );
            mustBePositive( N_points );
            mustBeInteger( N_points );

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( N_coefficients, N_points );

            %--------------------------------------------------------------
            % 2.) create linear transforms
            %--------------------------------------------------------------
            % repeat default linear transform
            objects = repmat( objects, size( N_coefficients ) );

            % iterate linear transforms
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).N_coefficients = N_coefficients( index_object );
                objects( index_object ).N_points = N_points( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = linear_transform( N_coefficients, N_points )

        %------------------------------------------------------------------
        % sizes of the linear transforms
        %------------------------------------------------------------------
        function y = size_transform( LTs )

            % return sizes of the linear transforms
            y = [ [ LTs.N_coefficients ]; [ LTs.N_points ] ];

        end % function y = size_transform( LTs )

        %------------------------------------------------------------------
        % forward transform
        %------------------------------------------------------------------
        function y = forward_transform( LTs, x )

        end % function y = forward_transform( LTs, x )

        %------------------------------------------------------------------
        % adjoint transform
        %------------------------------------------------------------------
        function y = adjoint_transform( LTs, x )

        end % function y = adjoint_transform( LTs, x )

        %------------------------------------------------------------------
        % transform operator
        %------------------------------------------------------------------
        function y = operator_transform( LT, x, mode )

            % check mode
            switch mode

                case 0

                    %------------------------------------------------------
                    % return size of forward transform
                    %------------------------------------------------------
                    y = size_transform( LT );

                case 1

                    %------------------------------------------------------
                    % forward transform
                    %------------------------------------------------------
                    y = forward_transform( LT, x );

                case 2

                    %------------------------------------------------------
                    % adjoint transform
                    %------------------------------------------------------
                    y = adjoint_transform( LT, x );

                otherwise

                    %------------------------------------------------------
                    % invalid mode
                    %------------------------------------------------------
                    errorStruct.message = sprintf( 'Mode %d is invalid!', mode );
                    errorStruct.identifier = 'operator_transform:InvalidMode';
                    error( errorStruct );

            end % switch mode

        end % function y = operator_transform( LT, x, mode )

    end % methods
    
end % classdef linear_transform
