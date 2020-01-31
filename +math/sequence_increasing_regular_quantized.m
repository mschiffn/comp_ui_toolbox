%
% superclass for all strictly monotonically increasing sequences w/
% regular spacing and quantized bounds
%
% author: Martin F. Schiffner
% date: 2019-03-29
% modified: 2020-01-21
%
classdef sequence_increasing_regular_quantized < math.sequence_increasing_regular

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        q_lb ( 1, 1 ) int64     % lower integer bound
        q_ub ( 1, 1 ) int64     % upper integer bound

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = sequence_increasing_regular_quantized( lbs_q, ubs_q, deltas )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure integers for lbs_q and ubs_q
            mustBeInteger( lbs_q );
            mustBeInteger( ubs_q );

            % ensure positive deltas
            % (superclass ensures class physical_values.physical_quantity)
            mustBePositive( deltas );

            % multiple lbs_q / single deltas
            if ~isscalar( lbs_q ) && isscalar( deltas )
                deltas = repmat( deltas, size( lbs_q ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( lbs_q, ubs_q, deltas );

            %--------------------------------------------------------------
            % 2.) compute strictly monotonically increasing members
            %--------------------------------------------------------------
            % compute offsets
            offsets = double( lbs_q ) .* deltas;

            % compute numbers of members
            N_members = double( ubs_q - lbs_q + 1 );

            % constructor of superclass
            objects@math.sequence_increasing_regular( offsets, deltas, N_members );

            % iterate sequences
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).q_lb = int64( lbs_q( index_object ) );
                objects( index_object ).q_ub = int64( ubs_q( index_object ) );

            end % for index_object = 1:numel( objects )

        end % function objects = sequence_increasing_regular_quantized( lbs_q, ubs_q, deltas )

        %------------------------------------------------------------------
        % interpolate
        %------------------------------------------------------------------
        function sequences = interpolate( sequences, factors_interp )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class math.sequence_increasing_regular_quantized
            if ~isa( sequences, 'math.sequence_increasing_regular_quantized' )
                errorStruct.message = 'sequences must be math.sequence_increasing_regular_quantized!';
                errorStruct.identifier = 'interpolate:NoRegularQuantizedSequence';
                error( errorStruct );
            end

            % ensure equal subclasses of physical_values.physical_quantity
            auxiliary.mustBeEqualSubclasses( 'physical_values.physical_quantity', sequences.delta );

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
            % numbers of samples and sampling parameters
            lbs_q = reshape( [ sequences.q_lb ], size( sequences ) );
            ubs_q = reshape( [ sequences.q_ub ], size( sequences ) );
            deltas = reshape( [ sequences.delta ], size( sequences ) );

            % create axes for interpolated signal matrices
            lbs_q_int = double( lbs_q ) .* factors_interp;
            ubs_q_int = double( ubs_q + 1 ) .* factors_interp - 1;
            deltas_int = deltas ./ factors_interp;

            % create interpolated sequences
            sequences = math.sequence_increasing_regular_quantized( lbs_q_int, ubs_q_int, deltas_int );

        end % function sequences = interpolate( sequences, factors_interp )

        %------------------------------------------------------------------
        % cut out subsequence
        %------------------------------------------------------------------
        function [ sequences, indicators ] = cut_out( sequences, lbs, ubs )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass method cut_out ensures correct arguments

            %--------------------------------------------------------------
            % 2.) perform cut out
            %--------------------------------------------------------------
            % call cut out method in superclass
            [ sequences, indicators ] = cut_out@math.sequence_increasing( sequences, lbs, ubs );

            % ensure cell array for indicators
            if ~iscell( indicators )
                indicators = { indicators };
            end

            % iterate sequences
            for index_object = 1:numel( sequences )

                % find indices and values of nonzero elements
                indices = find( indicators{ index_object } );

                % update lower and upper integer bounds
                sequences( index_object ).q_lb = sequences( index_object ).q_lb + indices( 1 ) - 1;
                sequences( index_object ).q_ub = sequences( index_object ).q_lb + numel( indices ) - 1;

            end % for index_object = 1:numel( sequences )

            % avoid cell array for single sequence
            if isscalar( sequences )
                indicators = indicators{ 1 };
            end

        end % function [ sequences, indicators ] = cut_out( sequences, lbs, ubs )

        %------------------------------------------------------------------
        % remove last members
        %------------------------------------------------------------------
        function [ sequences, N_remove ] = remove_last( sequences, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass method remove_last ensures correct arguments

            %--------------------------------------------------------------
            % 2.) remove last members
            %--------------------------------------------------------------
            % call remove_last method in superclass
            [ sequences, N_remove ] = remove_last@math.sequence_increasing( sequences, varargin{ : } );

            % iterate sequences
            for index_object = 1:numel( sequences )

                % update upper integer bounds
                sequences( index_object ).q_ub = sequences( index_object ).q_ub - N_remove( index_object );

            end % for index_object = 1:numel( sequences )

        end % function [ sequences, N_remove ] = remove_last( sequences, varargin )

    end % methods

end % classdef sequence_increasing_regular_quantized < math.sequence_increasing_regular
