%
% superclass for all linear transforms
%
% abstract superclass for all linear transforms
%
% author: Martin F. Schiffner
% date: 2016-08-12
% modified: 2020-10-08
%
classdef (Abstract) linear_transform

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        N_coefficients ( 1, 1 ) double { mustBePositive, mustBeInteger, mustBeNonempty } = 1	% number of rows
        N_points ( 1, 1 ) double { mustBePositive, mustBeInteger, mustBeNonempty } = 1          % number of columns

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
            % ensure two arguments
            narginchk( 2, 2 );

            % property validation functions ensure nonempty positive integers for N_coefficients
            % property validation functions ensure nonempty positive integers for N_points

            % ensure equal number of dimensions and sizes
            [ N_coefficients, N_points ] = auxiliary.ensureEqualSize( N_coefficients, N_points );

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
        % transform operator
        %------------------------------------------------------------------
        function y = operator_transform( LTs, x, mode )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.linear_transform
            if ~isa( LTs, 'linear_transforms.linear_transform' )
                errorStruct.message = 'LTs must be linear_transforms.linear_transform!';
                errorStruct.identifier = 'operator_transform:NoLinearTransforms';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) apply transform operator
            %--------------------------------------------------------------
            % check mode
            switch mode

                case 0

                    %------------------------------------------------------
                    % return size of forward transform
                    %------------------------------------------------------
                    y = size_transform( LTs );

                case 1

                    %------------------------------------------------------
                    % forward transform
                    %------------------------------------------------------
                    y = forward_transform( LTs, x );

                case 2

                    %------------------------------------------------------
                    % adjoint transform
                    %------------------------------------------------------
                    y = adjoint_transform( LTs, x );

                otherwise

                    %------------------------------------------------------
                    % invalid mode
                    %------------------------------------------------------
                    errorStruct.message = sprintf( 'Mode %d is invalid!', mode );
                    errorStruct.identifier = 'operator_transform:InvalidMode';
                    error( errorStruct );

            end % switch mode

        end % function y = operator_transform( LTs, x, mode )

    end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (Abstract)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract)

        %------------------------------------------------------------------
        % forward transform
        %------------------------------------------------------------------
        y = forward_transform( LTs, x )

        %------------------------------------------------------------------
        % adjoint transform
        %------------------------------------------------------------------
        y = adjoint_transform( LTs, x )

        %------------------------------------------------------------------
        % display coefficients
        %------------------------------------------------------------------
        display_coefficients( LTs, x )

	end % methods (Abstract)

end % classdef (Abstract) linear_transform
