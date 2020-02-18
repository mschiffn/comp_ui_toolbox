%
% superclass for all sequence reweighting options
%
% author: Martin F. Schiffner
% date: 2019-09-17
% modified: 2020-02-12
%
classdef sequence < regularization.options.reweightings.reweighting

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        q ( 1, 1 ) double { mustBeNonnegative, mustBeLessThanOrEqual( q, 2 ) } = 0.5	% norm parameter
        epsilon_n ( :, 1 ) double { mustBeNonnegative } = 1 ./ ( 1 + (1:5) )            % reweighting sequence

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = sequence( q, epsilon_n )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % property validation functions ensure valid q

            % ensure cell array for epsilon_n
            if ~iscell( epsilon_n )
                epsilon_n = { epsilon_n };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( q, epsilon_n );

            %--------------------------------------------------------------
            % 2.) create sequence reweighting options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.options.reweightings.reweighting( size( q ) );

            % iterate sequence reweighting options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).q = q( index_object );
                objects( index_object ).epsilon_n = epsilon_n{ index_object };

            end % for index_object = 1:numel( objects )

        end % function objects = sequence( epsilon_n )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( options_sequence )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.options.reweightings.sequence
            if ~isa( options_sequence, 'regularization.options.reweightings.sequence' )
                errorStruct.message = 'options_sequence must be regularization.options.reweightings.sequence!';
                errorStruct.identifier = 'string:NoOptionsReweightingSequence';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % initialize string array for strs_out
            strs_out = repmat( "", size( options_sequence ) );

            % iterate sequence reweighting options
            for index_object = 1:numel( options_sequence )

                strs_out( index_object ) = sprintf( "%s (q = %2.1f, N = %d)", 'sequence', options_sequence( index_object ).q, numel( options_sequence( index_object ).epsilon_n ) );

            end % for index_object = 1:numel( options_sequence )

        end % function strs_out = string( options_sequence )

	end % methods

end % classdef sequence < regularization.options.reweightings.reweighting
