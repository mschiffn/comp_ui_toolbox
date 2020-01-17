%
% superclass for all strictly monotonically increasing sequences w/
% regular spacing
%
% author: Martin F. Schiffner
% date: 2020-01-11
% modified: 2020-01-16
%
classdef sequence_increasing_regular < math.sequence_increasing

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        offset ( 1, 1 ) physical_values.physical_quantity { mustBeNonempty } = physical_values.second( 0 )                  % arbitrary offset
        delta ( 1, 1 ) physical_values.physical_quantity { mustBePositive, mustBeNonempty } = physical_values.second( 1 )	% step size

    end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = sequence_increasing_regular( offsets, deltas, N_members )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure equal subclasses of physical_values.physical_quantity
            auxiliary.mustBeEqualSubclasses( 'physical_values.physical_quantity', offsets, deltas );

            % property validation function ensures positive integers for N_members

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( offsets, deltas, N_members );

            %--------------------------------------------------------------
            % 2.) create strictly monotonically increasing sequences (regular spacing)
            %--------------------------------------------------------------
            % specify cell array for members
            members = cell( size( offsets ) );

            % iterate sequences
            for index_object = 1:numel( offsets )

                % compute members
                members{ index_object } = offsets( index_object ) + ( 0:( N_members( index_object ) - 1 ) )' * deltas( index_object );

            end % for index_object = 1:numel( offsets )

            % constructor of superclass
            objects@math.sequence_increasing( members );

            % iterate sequences
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).offset = offsets( index_object );
                objects( index_object ).delta = deltas( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = sequence_increasing_regular( offsets, deltas, N_members )

        %------------------------------------------------------------------
        % interpolate
        %------------------------------------------------------------------
        function sequences = interpolate( sequences, factors_interp )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class math.sequence_increasing_regular
            if ~isa( sequences, 'math.sequence_increasing_regular' )
                errorStruct.message = 'sequences must be math.sequence_increasing_regular!';
                errorStruct.identifier = 'interpolate:NoRegularQuantizedSequence';
                error( errorStruct );
            end

            % ensure equal subclasses of physical_values.physical_quantity
            auxiliary.mustBeEqualSubclasses( 'physical_values.physical_quantity', sequences.members );

            % ensure positive integers
            mustBePositive( factors_interp );
            mustBeInteger( factors_interp );

            % multiple sequences / single factors_interp
            if ~isscalar( sequences ) && isscalar( factors_interp )
                factors_interp = repmat( factors_interp, size( sequences ) );
            end

            % single sequences / multiple factors_interp
            if isscalar( sequences ) && ~isscalar( factors_interp )
                sequences = repmat( sequences, size( factors_interp ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( sequences, factors_interp );

            %--------------------------------------------------------------
            % 2.) interpolate sequences
            %--------------------------------------------------------------
            % extract offsets and deltas
            offsets = reshape( [ sequences.offset ], size( sequences ) );
            deltas = reshape( [ sequences.delta ], size( sequences ) );

            % interpolate deltas / new numbers of members
            deltas_int = deltas ./ factors_interp;
            N_members_int = abs( sequences ) .* factors_interp;

            % create strictly monotonically increasing sequences w/ regular spacing
            sequences = math.sequence_increasing_regular( offsets, deltas_int, N_members_int );

        end % function sequences = interpolate( sequences, factors_interp )

    end % methods

end % classdef sequence_increasing_regular < math.sequence_increasing
